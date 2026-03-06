# orchestrator/ai_client.py (new version using AWS Bedrock Claude Sonnet 4.5)

import json
import boto3
from typing import List, Optional

class ClaudeBedrockClient:
    """
    Simple wrapper for AWS Bedrock Claude Sonnet 4.5.
    """

    def __init__(
        self,
        inference_profile_id: str,
        model_id: str = "anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name: str = "us-east-1",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        self.inference_profile_id = inference_profile_id
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_messages: Optional[List[dict]] = None,
    ) -> str:
        """
        extra_messages: optional list of {"role": "user"/"assistant", "content": "..."}
        """
        messages = []
        if extra_messages:
            messages.extend(extra_messages)

        messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": [{"type": "text","text": system_prompt}],
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = self.client.invoke_model(
            modelId=self.inference_profile_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        response_body = json.loads(response["body"].read())
        # Anthropic-style messages: response_body["output"]["message"]["content"]
        content_blocks = response_body["content"]
        # Find first text block
        for block in content_blocks:
            if block["type"] == "text":
                print('ai response', block)
                return block["text"]
        return ""
