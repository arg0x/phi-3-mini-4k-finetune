from ctransformers import AutoModelForCausalLM
import gradio as gr

llm = AutoModelForCausalLM.from_pretrained("phi-3-mini-4k-emotional-support-gguf-unsloth.Q4_K_M.gguf",
max_new_tokens = 1096,
threads = 3,
)

def stream(prompt, UL):
    # Add your prompt here
    system_prompt = ''
    E_INST = "</s>"
    user, assistant = "<|user|>", "<|assistant|>"
    prompt = f"{system_prompt}{E_INST}\n{user}\n{prompt.strip()}{E_INST}\n{assistant}\n"
    return llm(prompt)

css = """
  h1 {
  text-align: center;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
.contain {
  max-width: 900px;
  margin: auto;
  padding-top: 1.5rem;
}
"""

chat_interface = gr.ChatInterface(
    fn=stream,
    stop_btn=None,
    examples=[
        ["How do I manage stress?"]
    ],
)

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1><center>Phi-3-mini Emotional Support<h1><center>")
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chat_interface.render()
    
if __name__ == "__main__":
    demo.queue(max_size=10).launch()