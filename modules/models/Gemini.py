import requests
import json
from ..presets import *
from ..utils import *

from .base_model import BaseLLMModel

class GeminiClient(BaseLLMModel):
    @staticmethod
    def construct_gemini_text(role, text):
        return {"role": role, "parts": [{"text": text}]}
    def __init__(self, model_name, api_key, username):
        super().__init__(model_name=model_name)
        self.model_name = model_name
        self.api_key = api_key
        self.user_name = username
        if self.api_key == None:
            raise Exception("请在配置文件或者环境变量中设置Gemini Key")
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.gemini_stream_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent"
    
    def add_user_input(self, text):
        """
        向历史记录中添加用户输入。
        """
        self.history.append({"role": "user", "parts": [{"text": text}]})

    def get_answer_stream_iter(self):
        """
        以流式方式获取答案。
        """
        history = []
        if self.system_prompt is not None:
            history = [self.construct_gemini_text("system", self.system_prompt)]
        for i in self.history:
            history.append(self.construct_gemini_text(i["role"], i["content"]))
        payload = json.dumps({"contents": self.history})
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_key
        }

        response = requests.post(self.gemini_stream_url + "?key=" + self.api_key, headers=headers, data=payload, stream=True)

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_content = json.loads(line.decode('utf-8'))
                    yield line_content.get("text", "")
        else:
            yield "请求失败，状态码：{}".format(response.status_code)

    def get_answer_at_once(self):
        """
        一次性获取完整答案。
        """
        system_prompt = self.system_prompt
        history = self.history
        if system_prompt is not None:
            history = [self.construct_gemini_text("system", system_prompt), *self.history]
        payload = json.dumps({"contents": history})
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_key
        }

        response = requests.post(self.gemini_url + "?key=" + self.api_key, headers=headers, data=payload)

        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""), len(response.content)
        else:
            return "请求失败，状态码：{}".format(response.status_code), 0