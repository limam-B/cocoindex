use crate::prelude::*;

use super::{LlmEmbeddingClient, LlmGenerationClient, LlmProgressUpdate};
use futures::StreamExt;
use schemars::schema::SchemaObject;
use serde_with::{base64::Base64, serde_as};

fn get_embedding_dimension(model: &str) -> Option<u32> {
    match model.to_ascii_lowercase().as_str() {
        "mxbai-embed-large"
        | "bge-m3"
        | "bge-large"
        | "snowflake-arctic-embed"
        | "snowflake-arctic-embed2" => Some(1024),

        "nomic-embed-text"
        | "paraphrase-multilingual"
        | "snowflake-arctic-embed:110m"
        | "snowflake-arctic-embed:137m"
        | "granite-embedding:278m" => Some(768),

        "all-minilm"
        | "snowflake-arctic-embed:22m"
        | "snowflake-arctic-embed:33m"
        | "granite-embedding" => Some(384),

        _ => None,
    }
}

pub struct Client {
    generate_url: String,
    embed_url: String,
    reqwest_client: reqwest::Client,
}

#[derive(Debug, Serialize)]
enum OllamaFormat<'a> {
    #[serde(untagged)]
    JsonSchema(&'a SchemaObject),
}

#[serde_as]
#[derive(Debug, Serialize)]
struct OllamaRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    #[serde_as(as = "Option<Vec<Base64>>")]
    pub images: Option<Vec<&'a [u8]>>,
    pub format: Option<OllamaFormat<'a>>,
    pub system: Option<&'a str>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    pub response: String,
}

/// Streaming response from Ollama (one chunk at a time)
#[derive(Debug, Deserialize)]
struct OllamaStreamResponse {
    pub response: String,
    #[serde(default)]
    pub done: bool,
}

#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequest<'a> {
    pub model: &'a str,
    pub input: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
}

const OLLAMA_DEFAULT_ADDRESS: &str = "http://localhost:11434";

impl Client {
    pub async fn new(address: Option<String>) -> Result<Self> {
        let address = match &address {
            Some(addr) => addr.trim_end_matches('/'),
            None => OLLAMA_DEFAULT_ADDRESS,
        };
        Ok(Self {
            generate_url: format!("{address}/api/generate"),
            embed_url: format!("{address}/api/embed"),
            reqwest_client: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl LlmGenerationClient for Client {
    async fn generate<'req>(
        &self,
        request: super::LlmGenerateRequest<'req>,
    ) -> Result<super::LlmGenerateResponse> {
        // Check if we have a progress callback for streaming
        let use_streaming = request.progress_callback.is_some();

        let req = OllamaRequest {
            model: request.model,
            prompt: request.user_prompt.as_ref(),
            images: request.image.as_deref().map(|img| vec![img]),
            format: request.output_format.as_ref().map(
                |super::OutputFormat::JsonSchema { schema, .. }| {
                    OllamaFormat::JsonSchema(schema.as_ref())
                },
            ),
            system: request.system_prompt.as_ref().map(|s| s.as_ref()),
            stream: Some(use_streaming),
        };

        if use_streaming {
            // Streaming mode with progress callback
            let progress_callback = request.progress_callback.unwrap();
            let response = self
                .reqwest_client
                .post(self.generate_url.as_str())
                .json(&req)
                .send()
                .await
                .context("Ollama API error")?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                bail!("Ollama API error: {} - {}", status, error_text);
            }

            let mut full_response = String::new();
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.context("Error reading stream chunk")?;

                // Parse each line as a JSON object (Ollama sends newline-delimited JSON)
                for line in chunk.split(|&b| b == b'\n') {
                    if line.is_empty() {
                        continue;
                    }

                    if let Ok(stream_resp) = serde_json::from_slice::<OllamaStreamResponse>(line) {
                        full_response.push_str(&stream_resp.response);

                        // Call progress callback with the chunk
                        progress_callback(LlmProgressUpdate {
                            text_chunk: stream_resp.response,
                            done: stream_resp.done,
                        });
                    }
                }
            }

            Ok(super::LlmGenerateResponse {
                text: full_response,
            })
        } else {
            // Non-streaming mode (original behavior)
            let res = http::request(|| {
                self.reqwest_client
                    .post(self.generate_url.as_str())
                    .json(&req)
            })
            .await
            .context("Ollama API error")?;
            let json: OllamaResponse = res.json().await?;
            Ok(super::LlmGenerateResponse {
                text: json.response,
            })
        }
    }

    fn json_schema_options(&self) -> super::ToJsonSchemaOptions {
        super::ToJsonSchemaOptions {
            fields_always_required: false,
            supports_format: true,
            extract_descriptions: true,
            top_level_must_be_object: false,
            supports_additional_properties: true,
        }
    }
}

#[async_trait]
impl LlmEmbeddingClient for Client {
    async fn embed_text<'req>(
        &self,
        request: super::LlmEmbeddingRequest<'req>,
    ) -> Result<super::LlmEmbeddingResponse> {
        let texts: Vec<&str> = request.texts.iter().map(|t| t.as_ref()).collect();
        let req = OllamaEmbeddingRequest {
            model: request.model,
            input: texts,
        };
        let resp = http::request(|| self.reqwest_client.post(self.embed_url.as_str()).json(&req))
            .await
            .context("Ollama API error")?;

        let embedding_resp: OllamaEmbeddingResponse = resp.json().await.context("Invalid JSON")?;

        Ok(super::LlmEmbeddingResponse {
            embeddings: embedding_resp.embeddings,
        })
    }

    fn get_default_embedding_dimension(&self, model: &str) -> Option<u32> {
        get_embedding_dimension(model)
    }
}
