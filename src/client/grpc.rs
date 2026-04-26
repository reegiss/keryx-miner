use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockTemplateRequestMessage, GetInfoRequestMessage, KaspadMessage, NotifyBlockAddedRequestMessage,
    NotifyNewBlockTemplateRequestMessage,
};
use crate::{miner::MinerManager, Error};
use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use semver::Version;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc::{self, error::SendError, Sender}, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::Channel as TonicChannel, Streaming};

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"), "/", env!("PACKAGE_COMPILE_TIME"));
static VERSION_UPDATE: &str = "0.11.15";
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

    /// Pending AI response tag to embed in the next coinbase extra_data.
    /// Format: "ai:r:{payload_prefix_16}:{base64_result}"
    /// Set when an AiRequest TX is detected in the current block template.
    /// Consumed (and cleared) on the next GetBlockTemplate call.
    pending_ai_response: Option<String>,

    /// In-flight SLM inference task: (payload_prefix, result_receiver).
    /// Polled each time a new block template is requested.
    inference_rx: Option<(String, oneshot::Receiver<String>)>,
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
            pending_ai_response: None,
            inference_rx: None,
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
        // Unique nonce per request → distinct coinbase payloads at the same blue_score → no DAG tx_id collisions.
        let nonce_hex = format!("{:016x}", thread_rng().next_u64());
        // OPoI Phase 1: run the deterministic MLP (matches node validation).
        let opoi_tag = keryx_miner::inference::compute_opoi_tag(&nonce_hex);
        // Append TinyLlama AI response if one is ready from a previous block template scan.
        // Format appended: "/ai:r:{payload_prefix_16}:{base64url_result}"
        let ai_response = self.pending_ai_response.take()
            .map(|r| format!("/{}", r))
            .unwrap_or_default();
        let extra_data = format!("{}/{}/ai:v1:{}{}", EXTRA_DATA, nonce_hex, opoi_tag, ai_response);
        self.client_send(GetBlockTemplateRequestMessage { pay_address, extra_data }).await
    }

    async fn handle_message(&mut self, msg: Payload, miner: &mut MinerManager) -> Result<(), Error> {
        match msg {
            Payload::BlockAddedNotification(_) => self.client_get_block_template().await?,
            Payload::NewBlockTemplateNotification(_) => self.client_get_block_template().await?,
            Payload::GetBlockTemplateResponse(template) => {
                // Poll the in-flight SLM inference task if one is running.
                if self.pending_ai_response.is_none() {
                    if let Some((ref prefix, ref mut rx)) = self.inference_rx {
                        if let Ok(result) = rx.try_recv() {
                            use base64::Engine as _;
                            // URL_SAFE_NO_PAD avoids '/' and '=' which would
                            // break the '/' delimiter used in extra_data parsing.
                            let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
                                .encode(result.as_bytes());
                            let tag = format!("ai:r:{}:{}", prefix, encoded);
                            info!("OPoI: inference complete, queued response {}", tag);
                            self.pending_ai_response = Some(tag);
                            self.inference_rx = None;
                        }
                    }
                }
                // Scan non-coinbase TXs for AiRequest payloads.
                // When found, kick off SLM inference in a blocking thread.
                if self.pending_ai_response.is_none() && self.inference_rx.is_none() {
                    if let Some(ref block) = template.block {
                        for tx in &block.transactions {
                            if tx.inputs.is_empty() {
                                continue; // skip coinbase
                            }
                            if let Some((prefix, prompt, max_tokens)) =
                                keryx_miner::inference::extract_ai_request(&tx.payload)
                            {
                                info!("OPoI: detected AiRequest (prefix={}), spawning SLM inference", prefix);
                                let (tx_done, rx_done) = oneshot::channel::<String>();
                                tokio::task::spawn_blocking(move || {
                                    let result = keryx_miner::slm::run_inference(&prompt, max_tokens);
                                    let _ = tx_done.send(result);
                                });
                                self.inference_rx = Some((prefix, rx_done));
                                break;
                            }
                        }
                    }
                }
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
            Payload::SubmitBlockResponse(res) => match res.error {
                None => info!("block submitted successfully!"),
                Some(e) => warn!("Failed submitting block: {:?}", e),
            },
            Payload::GetBlockResponse(msg) => {
                if let Some(e) = msg.error {
                    return Err(e.message.into());
                } else {
                    info!("Get block response: {:?}", msg);
                }
            }
            Payload::GetInfoResponse(info) => {
                info!("Keryxd version: {}", info.server_version);
                let keryxd_version = Version::parse(&info.server_version)?;
                let update_version = Version::parse(VERSION_UPDATE)?;
                match keryxd_version >= update_version {
                    true => self.client_send(NotifyNewBlockTemplateRequestMessage {}).await?,
                    false => self.client_send(NotifyBlockAddedRequestMessage {}).await?,
                };

                self.client_get_block_template().await?;
            }
            Payload::NotifyNewBlockTemplateResponse(res) => match res.error {
                None => info!("Registered for new template notifications"),
                Some(e) => error!("Failed registering for new template notifications: {:?}", e),
            },
            Payload::NotifyBlockAddedResponse(res) => match res.error {
                None => info!("Registered for block notifications (upgrade your Keryxd for better experience)"),
                Some(e) => error!("Failed registering for block notifications: {:?}", e),
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
