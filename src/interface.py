import gradio as gr


def create_interface(get_response):
    description = "Ищете идеального спутника для своих путешествий? Наш AI гид — это ваш надежный источник информации о туристических направлениях, культурных традициях и исторических фактах. Он анализирует контекст и предоставляет вам персонализированные рекомендации, чтобы вы могли максимально насладиться своим путешествием. Просто задайте вопрос, и откройте для себя мир новых возможностей!"

    interface = gr.Blocks(title="🌏🧳🛩️  AI Travel Guide", theme="citrus")

    with interface:
        gr.Markdown("# 🌏🧳🛩️  AI Travel Guide")
        gr.Markdown(description)
        with gr.Row():
            text_input = gr.Textbox()
        with gr.Row():
            text_button = gr.Button("Спросить AI гида")
        with gr.Row():
            text_output = gr.Markdown()

        text_button.click(get_response, inputs=text_input, outputs=text_output)

    return interface


if __name__ == "__main__":

    def mock_response(query):
        return """# AI GUIDE 
**Привет!** 
Я твой гид!"""

    interface = create_interface(mock_response)
    interface.launch(share=False, debug=True)
