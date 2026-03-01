import gradio as gr

from app.services import audio as audio_service
from app.services import chat as chat_service
from app.services import knowledge
from app.utils import history
from app.utils.logger import get_logger

log = get_logger(__name__)


# ── HTML content ─────────────────────────────────────────────────────

HOME_HTML = """
<div style="max-width:800px; margin:auto; padding:30px 20px; font-family:system-ui, sans-serif;">
  <div style="text-align:center; margin-bottom:30px;">
    <h1 style="font-size:2.4em; margin-bottom:5px;">MediAssist AI</h1>
    <p style="font-size:1.1em; opacity:0.8;">Your Intelligent Medical Assistance Companion</p>
  </div>

  <div style="background:var(--block-background-fill); padding:25px; border-radius:12px; margin-bottom:20px;">
    <h2 style="margin-top:0;">What is MediAssist?</h2>
    <p>MediAssist is an AI-powered medical chatbot that helps you with basic health queries,
       recommends over-the-counter remedies, and connects you to nearby hospitals when needed.
       It supports both <strong>English</strong> and <strong>Urdu</strong>.</p>
  </div>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-bottom:20px;">
    <div style="background:var(--block-background-fill); padding:20px; border-radius:12px;">
      <h3 style="margin-top:0;">Text Chat</h3>
      <p>Type your medical question in English or Urdu and get an informed response backed by medical literature.</p>
    </div>
    <div style="background:var(--block-background-fill); padding:20px; border-radius:12px;">
      <h3 style="margin-top:0;">Voice Chat</h3>
      <p>Speak your question using the microphone. The system transcribes, processes, and reads the answer back to you.</p>
    </div>
    <div style="background:var(--block-background-fill); padding:20px; border-radius:12px;">
      <h3 style="margin-top:0;">Knowledge Base</h3>
      <p>Upload medical PDFs to expand the chatbot's knowledge. The system indexes them for accurate retrieval.</p>
    </div>
    <div style="background:var(--block-background-fill); padding:20px; border-radius:12px;">
      <h3 style="margin-top:0;">Hospital Finder</h3>
      <p>Share your location and MediAssist will find nearby hospitals with Google Maps links.</p>
    </div>
  </div>
</div>
"""

ABOUT_HTML = """
<div style="max-width:800px; margin:auto; padding:30px 20px; font-family:system-ui, sans-serif;">
  <div style="text-align:center; margin-bottom:30px;">
    <h1 style="font-size:2.2em; margin-bottom:5px;">About MediAssist AI</h1>
  </div>

  <div style="background:var(--block-background-fill); padding:25px; border-radius:12px; margin-bottom:20px;">
    <h2 style="margin-top:0;">What It Does</h2>
    <p>MediAssist AI is a medical guidance chatbot that combines large language models with a
       retrieval-augmented generation (RAG) pipeline. It ingests medical PDFs, indexes them,
       and uses the knowledge to provide informed responses to health-related questions.</p>
    <ul>
      <li><strong>Bilingual:</strong> Supports English and Urdu (text and voice).</li>
      <li><strong>Voice-enabled:</strong> Speak your question, hear the answer via neural text-to-speech.</li>
      <li><strong>Knowledge-grounded:</strong> Responses are backed by indexed medical documents.</li>
      <li><strong>Hospital finder:</strong> Recommends nearby hospitals using Google Maps.</li>
      <li><strong>MLOps-ready:</strong> All interactions are tracked with MLflow for monitoring and improvement.</li>
    </ul>
  </div>

  <div style="background:var(--block-background-fill); padding:25px; border-radius:12px; margin-bottom:20px;">
    <h2 style="margin-top:0;">Quick Start Guide</h2>
    <ol>
      <li><strong>Upload PDFs</strong> &mdash; Go to the Knowledge Base tab and upload medical reference documents.</li>
      <li><strong>Ask a question</strong> &mdash; Type or speak your medical question in the Chat tab.</li>
      <li><strong>Get guidance</strong> &mdash; The chatbot responds using your uploaded knowledge base.</li>
      <li><strong>Find hospitals</strong> &mdash; If needed, share your location and the bot will find nearby hospitals.</li>
    </ol>
  </div>

</div>
"""


# ── Chat handlers ────────────────────────────────────────────────────

def _handle_audio(audio_input, chat_history: list):
    """Handle a microphone recording."""
    if audio_input is None:
        return chat_history, None

    try:
        result = audio_service.process_audio(audio_input)
    except Exception as exc:
        log.error("Audio error: %s", exc)
        chat_history = chat_history + [
            {"role": "user", "content": "(audio input)"},
            {"role": "assistant", "content": f"Sorry, an error occurred processing your audio: {exc}"},
        ]
        return chat_history, None

    stt_text = result["stt_text"]
    response_text = result["text_response"]
    audio_path = result.get("audio_path")

    chat_history = chat_history + [
        {"role": "user", "content": stt_text},
        {"role": "assistant", "content": response_text},
    ]
    return chat_history, audio_path


def _clear_chat():
    history.clear()
    return [], None


# ── Knowledge Base handlers ──────────────────────────────────────────

def _upload_pdf(file):
    if file is None:
        return "No file selected.\n\n" + _get_kb_data()
    result = knowledge.upload_pdf(file.name)
    return f"**{result['filename']}**: {result['status']} ({result['chunks']} chunks)\n\n" + _get_kb_data()


def _get_kb_data():
    docs = knowledge.list_documents()
    if not docs:
        return "No documents indexed yet."
    lines = ["| Filename | Chunks | Indexed At |", "| --- | --- | --- |"]
    for d in docs:
        lines.append(f"| `{d['filename']}` | {d.get('chunks', 0)} | {d.get('indexed_at', '')[:19]} |")
    return "\n".join(lines)


def _delete_doc(filename: str):
    if not filename:
        return "No document selected.\n\n" + _get_kb_data()
    success = knowledge.delete_document(filename.strip())
    msg = f"Deleted `{filename}`" if success else f"Could not find `{filename}`"
    return msg + "\n\n" + _get_kb_data()


def _rebuild():
    total = knowledge.rebuild_index()
    return f"Rebuilt index: {total} total chunks\n\n" + _get_kb_data()


# ── Build the Gradio app ────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="MediAssist AI") as app:

        # ── Home Tab ────────────────────────────────────
        with gr.Tab("Home"):
            gr.HTML(HOME_HTML)

        # ── Chat Tab ────────────────────────────────────
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=480, show_label=False)

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Type your question here ...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                mic_input = gr.Audio(
                    sources=["microphone"],
                    label="Or speak your question",
                    type="numpy",
                )
                audio_output = gr.Audio(
                    label="Response audio",
                    type="filepath",
                    interactive=False,
                    autoplay=True,
                )

            clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

            def _on_text_submit(user_text, history):
                if not user_text or not user_text.strip():
                    return history, ""
                history = history + [{"role": "user", "content": user_text}]
                try:
                    response = chat_service.process_message(user_text)
                except Exception as exc:
                    log.error("Chat error: %s", exc)
                    response = f"Sorry, an error occurred: {exc}"
                history = history + [{"role": "assistant", "content": response}]
                return history, ""

            chat_input.submit(
                _on_text_submit,
                [chat_input, chatbot],
                [chatbot, chat_input],
            )
            send_btn.click(
                _on_text_submit,
                [chat_input, chatbot],
                [chatbot, chat_input],
            )

            mic_input.stop_recording(
                _handle_audio,
                [mic_input, chatbot],
                [chatbot, audio_output],
            )

            clear_btn.click(_clear_chat, outputs=[chatbot, audio_output])

        # ── Knowledge Base Tab ──────────────────────────
        with gr.Tab("Knowledge Base"):
            gr.Markdown("### Manage your medical knowledge base\nUpload PDF documents to expand MediAssist's knowledge.")

            with gr.Row():
                pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("Index PDF", variant="primary")

            gr.Markdown("### Indexed Documents")
            kb_panel = gr.Markdown(value=_get_kb_data(), min_height=80)

            with gr.Row():
                delete_input = gr.Textbox(label="Filename to delete", placeholder="e.g. document.pdf")
                delete_btn = gr.Button("Delete", variant="stop")
                rebuild_btn = gr.Button("Rebuild Index", variant="secondary")

            upload_btn.click(_upload_pdf, [pdf_upload], [kb_panel])
            delete_btn.click(_delete_doc, [delete_input], [kb_panel])
            rebuild_btn.click(_rebuild, outputs=[kb_panel])

        # ── About Tab ──────────────────────────────────
        with gr.Tab("About"):
            gr.HTML(ABOUT_HTML)

    return app
