from advanced_chatbot import advanced_chat
import gradio as gr

def main():
    iface = gr.Interface(
        fn=advanced_chat,
        inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
        outputs="text",
        title="Advanced Conversational AI Chatbot with LoRA",
        description="Chatbot powered by RAG, enhanced with LoRA adapters and advanced fine-tuning stubs."
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
