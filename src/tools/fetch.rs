use anyhow::{anyhow, Context, Result};
use llm::builder::ParamBuilder;
use reqwest::blocking::{Client, Response};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{json, Value};
use std::str::FromStr;
use std::time::Duration;

use super::Tool;

pub struct FetchUrlTool;

impl Tool for FetchUrlTool {
    fn name(&self) -> &'static str {
        "fetch_url"
    }
    fn description(&self) -> &'static str {
        "Fetch content from an HTTP/HTTPS URL with optional method, headers, body, and timeout. Returns status, headers, and text (truncated)."
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("url")
                .type_of("string")
                .description("The URL to fetch (http or https)"),
            ParamBuilder::new("method")
                .type_of("string")
                .description("HTTP method (default GET)"),
            ParamBuilder::new("headers")
                .type_of("object")
                .description("Optional headers as key-value object"),
            ParamBuilder::new("body")
                .type_of("string")
                .description("Optional request body for POST/PUT/PATCH"),
            ParamBuilder::new("timeout_sec")
                .type_of("integer")
                .description("Request timeout in seconds (default 10)"),
            ParamBuilder::new("max_bytes")
                .type_of("integer")
                .description("Maximum response bytes to capture (default 200000)"),
        ]
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["url"]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'url'"))?;
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(anyhow!("Only http/https URLs are allowed"));
        }
        let method = args
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET")
            .to_uppercase();
        let timeout = args
            .get("timeout_sec")
            .and_then(|v| v.as_u64())
            .unwrap_or(10);
        let max_bytes = args
            .get("max_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(200_000) as usize;
        let body = args
            .get("body")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut headers = HeaderMap::new();
        if let Some(hv) = args.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in hv {
                if let Some(val_str) = v.as_str() {
                    if let Ok(name) = HeaderName::from_str(k) {
                        if let Ok(val) = HeaderValue::from_str(val_str) {
                            headers.insert(name, val);
                        }
                    }
                }
            }
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .connect_timeout(Duration::from_secs(timeout))
            .build()?;

        let req_builder = match method.as_str() {
            "GET" => client.get(url),
            "POST" => client.post(url),
            "PUT" => client.put(url),
            "PATCH" => client.patch(url),
            "DELETE" => client.delete(url),
            "HEAD" => client.head(url),
            _ => return Err(anyhow!("Unsupported method")),
        };
        let mut req = req_builder.headers(headers);
        if let Some(b) = body {
            req = req.body(b);
        }

        let resp: Response = req
            .send()
            .with_context(|| format!("Request failed for {}", url))?;
        let status = resp.status().as_u16();
        let final_url = resp.url().to_string();
        let mut resp_headers = serde_json::Map::new();
        for (name, value) in resp.headers().iter() {
            resp_headers.insert(name.to_string(), json!(value.to_str().unwrap_or("")));
        }
        let mut text = resp.text().unwrap_or_default();
        let truncated = text.len() > max_bytes;
        if truncated {
            text.truncate(max_bytes);
        }
        Ok(json!({
            "url": url,
            "final_url": final_url,
            "status": status,
            "headers": resp_headers,
            "truncated": truncated,
            "text": text,
        }))
    }
}
