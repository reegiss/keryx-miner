use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockRequestMessage, GetBlockTemplateRequestMessage, GetInfoRequestMessage, KaspadMessage,
    NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage,
};
use crate::{miner::MinerManager, Error};
use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc::{self, error::SendError, Sender}, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::Channel as TonicChannel, Streaming};

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"), "/", env!("PACKAGE_COMPILE_TIME"));
type BlockHandle = JoinHandle<Result<(), PollSendError<KaspadMessage>>>;

#[allow(dead_code)]
pub struct KeryxdHandler {
    client: RpcClient<TonicChannel>,
    pub send_channel: Sender<KaspadMessage>,
    stream: Streaming<KaspadMessage>,
    miner_address: String,
    mine_when_not_synced: bool,
    devfund_address: Option<String>,
    devfund_percent: u16,
    block_template_ctr: Arc<AtomicU16>,

    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// Queue of AiRequests seen in confirmed blocks or mempool, waiting for inference.
    /// Each entry: (payload_prefix_16, prompt, max_tokens).
    /// Fed by both BlockAdded scans and block template scans.
    ai_request_queue: VecDeque<(String, String, usize)>,

    /// Prefixes already queued or in-flight — used for deduplication.
    ai_seen_prefixes: std::collections::HashSet<String>,

    /// Pending AI response tag ready to embed in the next coinbase extra_data.
    /// Format: "ai:r:{payload_prefix_16}:{base64_result}"
    pending_ai_response: Option<String>,

    /// In-flight SLM inference task: (payload_prefix, result_receiver).
    inference_rx: Option<(String, oneshot::Receiver<String>)>,

    /// 64-char hex Schnorr pubkey for the OPoI escrow output.
    /// `Some` → 10% of this miner's blue-block rewards go to this key (recoverable).
    /// `None` → 10% is burned (standard miner).
    escrow_pubkey: Option<String>,
}

#[async_trait(?Send)]
impl Client for KeryxdHandler {
    fn add_devfund(&mut self, address: String, percent: u16) {
        self.devfund_address = Some(address);
        self.devfund_percent = percent;
    }

    async fn register(&mut self) -> Result<(), Error> {
        // We actually register in connect
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        while let Some(msg) = self.stream.message().await? {
            match msg.payload {
                Some(payload) => self.handle_message(payload, miner).await?,
                None => warn!("keryxd message payload is empty"),
            }
        }
        Ok(())
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl KeryxdHandler {
    pub async fn connect<D>(
        address: D,
        miner_address: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        escrow_pubkey: Option<String>,
    ) -> Result<Box<Self>, Error>
    where
        D: std::convert::TryInto<tonic::transport::Endpoint>,
        D::Error: Into<Error>,
    {
        let mut client = RpcClient::connect(address).await?;
        let (send_channel, recv) = mpsc::channel(2);
        send_channel.send(GetInfoRequestMessage {}.into()).await?;
        let stream = client.message_stream(ReceiverStream::new(recv)).await?.into_inner();
        let (block_channel, block_handle) = Self::create_block_channel(send_channel.clone());
        Ok(Box::new(Self {
            client,
            stream,
            send_channel,
            miner_address,
            mine_when_not_synced,
            devfund_address: None,
            devfund_percent: 0,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16))),
            block_channel,
            block_handle,
            ai_request_queue: VecDeque::new(),
            ai_seen_prefixes: std::collections::HashSet::new(),
            pending_ai_response: None,
            inference_rx: None,
            escrow_pubkey,
        }))
    }

    fn create_block_channel(send_channel: Sender<KaspadMessage>) -> (Sender<BlockSeed>, BlockHandle) {
        // KaspadMessage::submit_block(block)
        let (send, recv) = mpsc::channel::<BlockSeed>(1);
        (
            send,
            tokio::spawn(async move {
                ReceiverStream::new(recv)
                    .map(|block_seed| match block_seed {
                        FullBlock(block) => KaspadMessage::submit_block(*block),
                        PartialBlock { .. } => unreachable!("All blocks sent here should have arrived from here"),
                    })
                    .map(Ok)
                    .forward(PollSender::new(send_channel))
                    .await
            }),
        )
    }

    async fn client_send(&self, msg: impl Into<KaspadMessage>) -> Result<(), SendError<KaspadMessage>> {
        self.send_channel.send(msg.into()).await
    }

    async fn client_get_block_template(&mut self) -> Result<(), SendError<KaspadMessage>> {
        let pay_address = match &self.devfund_address {
            Some(devfund_address) if self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent => {
                devfund_address.clone()
            }
            _ => self.miner_address.clone(),
        };
        self.block_template_ctr.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000)).unwrap();
        // Append a per-request random nonce so that parallel blocks at the same blue_score
        // get distinct coinbase payloads → distinct tx_ids (avoids DAG coinbase collisions).
        let nonce_hex = format!("{:016x}", thread_rng().next_u64());
        // OPoI Phase 2: run the deterministic fixed-point MLP (matches node validation).
        let opoi_tag = keryx_miner::inference::compute_opoi_tag(&nonce_hex);
        // Embed the pending AI response in every template request until a block is
        // successfully submitted. Using clone (not take) so the response survives
        // template rotations — at 10 BPS most templates are replaced before the
        // miner finds a valid nonce, so a single take() would silently drop responses.
        let ai_response = self.pending_ai_response
            .as_deref()
            .map(|r| format!("/{}", r))
            .unwrap_or_default();
        // OPoI escrow: if a Schnorr pubkey is configured, embed it so the node can route
        // the 10% escrow cut to a recoverable output instead of burning it.
        let escrow_part = self.escrow_pubkey
            .as_deref()
            .map(|pk| format!("/escrow:{}", pk))
            .unwrap_or_default();
        let extra_data = format!("{}{}/{}/ai:v1:{}{}", EXTRA_DATA, escrow_part, nonce_hex, opoi_tag, ai_response);
        self.client_send(GetBlockTemplateRequestMessage { pay_address, extra_data }).await
    }

    /// Scans a slice of transactions for AiRequest payloads and pushes new
    /// entries into `ai_request_queue` (deduplication by prefix).
    fn scan_txs_for_ai_requests(&mut self, txs: &[crate::proto::RpcTransaction]) {
        for tx in txs {
            // Only process SUBNETWORK_ID_AI_REQUEST transactions (0x03…).
            if tx.subnetwork_id != keryx_inference::SUBNETWORK_ID_AI_REQUEST_HEX {
                continue;
            }
            if let Some((prefix, prompt, max_tokens)) =
                keryx_miner::inference::extract_ai_request(&tx.payload)
            {
                if !self.ai_seen_prefixes.contains(&prefix) {
                    info!("OPoI: queued AiRequest prefix={}", prefix);
                    self.ai_seen_prefixes.insert(prefix.clone());
                    self.ai_request_queue.push_back((prefix, prompt, max_tokens));
                }
            }
        }
    }

    /// Starts SLM inference for the next queued AiRequest, if no inference is
    /// already in flight and a response slot is free.
    fn try_start_inference(&mut self) {
        if self.pending_ai_response.is_some() || self.inference_rx.is_some() {
            return;
        }
        if let Some((prefix, prompt, max_tokens)) = self.ai_request_queue.pop_front() {
            info!("OPoI: spawning SLM inference for prefix={}", prefix);
            let (tx_done, rx_done) = oneshot::channel::<String>();
            tokio::task::spawn_blocking(move || {
                let result = keryx_miner::slm::run_inference(&prompt, max_tokens);
                let _ = tx_done.send(result);
            });
            self.inference_rx = Some((prefix, rx_done));
        }
    }

    /// Polls the in-flight inference task and promotes the result to
    /// `pending_ai_response` when ready.
    fn poll_inference(&mut self) {
        if self.pending_ai_response.is_some() {
            return;
        }
        if let Some((ref prefix, ref mut rx)) = self.inference_rx {
            if let Ok(result) = rx.try_recv() {
                use base64::Engine as _;
                let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .encode(result.as_bytes());
                let tag = format!("ai:r:{}:{}", prefix, encoded);
                info!("OPoI: inference complete, queued response {}", tag);
                self.pending_ai_response = Some(tag);
                self.inference_rx = None;
            }
        }
    }

    async fn handle_message(&mut self, msg: Payload, miner: &mut MinerManager) -> Result<(), Error> {
        match msg {
            // BlockAdded: scan confirmed block for AiRequests that may have been
            // confirmed before the miner saw them in the mempool.
            // Do NOT trigger a new block template here — NewBlockTemplate handles that.
            Payload::BlockAddedNotification(notif) => {
                if let Some(block) = notif.block {
                    if !block.transactions.is_empty() {
                        // Full block included — scan directly.
                        self.scan_txs_for_ai_requests(&block.transactions.clone());
                        self.try_start_inference();
                    } else {
                        // Transactions absent — fetch the full block from the node.
                        let hash = block
                            .verbose_data
                            .as_ref()
                            .map(|v| v.hash.clone())
                            .unwrap_or_default();
                        if !hash.is_empty() {
                            self.client_send(GetBlockRequestMessage {
                                hash,
                                include_transactions: true,
                            })
                            .await?;
                        }
                    }
                }
            }
            Payload::NewBlockTemplateNotification(_) => self.client_get_block_template().await?,
            Payload::GetBlockTemplateResponse(template) => {
                // Poll any in-flight inference and scan mempool TXs.
                self.poll_inference();
                if let Some(ref block) = template.block {
                    self.scan_txs_for_ai_requests(&block.transactions.clone());
                }
                self.try_start_inference();
                match (template.block, template.is_synced, template.error) {
                    (Some(b), true, None) => miner.process_block(Some(FullBlock(Box::new(b)))).await?,
                    (Some(b), false, None) if self.mine_when_not_synced => {
                        miner.process_block(Some(FullBlock(Box::new(b)))).await?
                    }
                    (_, false, None) => miner.process_block(None).await?,
                    (_, _, Some(e)) => {
                        return Err(format!("GetTemplate returned with an error: {:?}", e).into());
                    }
                    (None, true, None) => error!("No block and No Error!"),
                }
            }
            // GetBlock response: arrives after we requested a full block from BlockAdded.
            // Scan its transactions for AiRequests.
            Payload::GetBlockResponse(msg) => {
                if let Some(e) = msg.error {
                    warn!("GetBlockResponse error: {}", e.message);
                } else if let Some(block) = msg.block {
                    self.scan_txs_for_ai_requests(&block.transactions.clone());
                    self.try_start_inference();
                }
            }
            Payload::SubmitBlockResponse(res) => match res.error {
                None => {
                    info!("block submitted successfully!");
                    // Clear the pending response — the block carrying it is now on-chain.
                    // The next queued AiRequest (if any) will start on the next template.
                    if self.pending_ai_response.take().is_some() {
                        info!("OPoI: response committed on-chain, slot cleared");
                        self.try_start_inference();
                    }
                }
                Some(e) => warn!("Failed submitting block: {:?}", e),
            },
            Payload::GetInfoResponse(info) => {
                info!("Keryxd version: {}", info.server_version);
                // Register for both notification types:
                // - NewBlockTemplate drives the mining loop
                // - BlockAdded lets us scan confirmed blocks for AiRequests
                //   that were confirmed before the miner saw them in mempool
                self.client_send(NotifyNewBlockTemplateRequestMessage {}).await?;
                self.client_send(NotifyBlockAddedRequestMessage {}).await?;
                self.client_get_block_template().await?;
            }
            Payload::NotifyNewBlockTemplateResponse(res) => match res.error {
                None => info!("Registered for new template notifications"),
                Some(e) => error!("Failed registering for new template notifications: {:?}", e),
            },
            Payload::NotifyBlockAddedResponse(res) => match res.error {
                None => info!("Registered for block added notifications (AI request scanning)"),
                Some(e) => error!("Failed registering for block added notifications: {:?}", e),
            },
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
    }
}

impl Drop for KeryxdHandler {
    fn drop(&mut self) {
        self.block_handle.abort();
    }
}
