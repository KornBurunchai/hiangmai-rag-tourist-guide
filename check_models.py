from groq import Groq

client = Groq(api_key="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")

try:
    models = client.models.list()
    print("โมเดลที่บัญชีของคุณสามารถใช้ได้:")
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print("Error:", e)