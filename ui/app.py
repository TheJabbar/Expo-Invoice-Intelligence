import gradio as gr
import requests
import json
import os
from PIL import Image

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

def process_invoice(file):
    try:
        # Display the uploaded image
        if file:
            img = Image.open(file)
            image_display = img
        else:
            image_display = None

        with open(file, "rb") as f:
            files = {"file": f}
            resp = requests.post(f"{API_BASE_URL}/api/invoice/upload", files=files, timeout=300)

        if resp.status_code != 200:
            return [image_display, "ERROR", "ERROR", "", "", "", "", "", "", "", f"API Error: {resp.status_code}"]

        result = resp.json()

        # New API response format
        fields = {
            "vendor": result.get("vendor"),
            "invoice_no": result.get("invoice_no"),
            "invoice_date": result.get("invoice_date"),
            "tax": result.get("tax"),
            "total": result.get("total"),
            "debit_account": result.get("debit_account"),
            "credit_account": result.get("credit_account", "Accounts Payable")  # Default value
        }
        overall_confidence = result.get("confidence", 0)

        # Get individual field confidences from the response
        confs = result.get("field_confidences", {})
        if not confs:
            # Fallback to using overall confidence for all fields if individual confidences not available
            confs = {field: overall_confidence for field in fields.keys()}

        # Build UI components with confidence indicators
        components = []
        for field in ["vendor", "invoice_no", "invoice_date", "tax", "total", "debit_account", "credit_account"]:
            value = fields.get(field, "")
            conf = confs.get(field, 0)
            label = f"{field.replace('_', ' ').title()} (conf: {conf:.2f})"
            components.append(gr.Textbox(value=str(value) if value else "", label=label))

        # Determine status based on overall confidence (threshold of 0.75)
        status = "✅ Auto-approved" if overall_confidence >= 0.75 else "⚠️ Requires review"
        return [
            image_display,
            result["invoice_id"],
            result["prediction_id"],  # Using prediction_id for feedback
            *components,
            status
        ]
    except Exception as e:
        return [None, "ERROR", "ERROR", "", "", "", "", "", "", "", f"Processing error: {str(e)}"]

def submit_correction(invoice_id, vendor, invoice_no, invoice_date, tax, total, debit_account, credit_account):
    try:
        corrected = {}
        for field, value in [("vendor", vendor), ("invoice_no", invoice_no), ("invoice_date", invoice_date),
                           ("tax", tax), ("total", total), ("debit_account", debit_account), ("credit_account", credit_account)]:
            if value and value.strip() and value != "None":
                corrected[field] = value.strip()

        if not corrected:
            return "❌ No corrections submitted"

        resp = requests.post(
            f"{API_BASE_URL}/api/invoice/feedback",
            data={
                "invoice_id": invoice_id,  # Using the invoice_id instead of prediction_id
                "corrected_fields": json.dumps(corrected),
                "user_id": "user_demo"
            },
            timeout=300
        )

        if resp.status_code == 200:
            return "✅ Correction saved! Model learns and rebuilds every week."
        else:
            return f"❌ Save failed: {resp.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="Invoice Intelligence") as demo:
    gr.Markdown("# 📄 Invoice Intelligence - Human-in-the-Loop OCR")
    gr.Markdown("Upload an invoice to extract fields. Low-confidence fields require correction.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Invoice (PDF/Image)", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            status_display = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            # Image preview component
            image_preview = gr.Image(label="Invoice Preview", interactive=False)

    invoice_id_state = gr.State("")
    prediction_id_state = gr.State("")

    with gr.Row():
        vendor = gr.Textbox(label="Vendor (conf: 0.00)")
        invoice_no = gr.Textbox(label="Invoice No (conf: 0.00)")

    with gr.Row():
        invoice_date = gr.Textbox(label="Invoice Date (conf: 0.00)")
        total = gr.Textbox(label="Total Amount (conf: 0.00)")

    with gr.Row():
        tax = gr.Textbox(label="Tax Amount (conf: 0.00)")
        debit_account = gr.Textbox(label="Debit Account (conf: 0.00)")

    with gr.Row():
        credit_account = gr.Textbox(label="Credit Account (conf: 0.00)")

    submit_btn = gr.Button("Submit Corrections", variant="primary", size="lg")
    result = gr.Textbox(label="Result", interactive=False)

    file_input.change(
        process_invoice,
        inputs=[file_input],
        outputs=[image_preview, invoice_id_state, prediction_id_state, vendor, invoice_no, invoice_date,
                tax, total, debit_account, credit_account, status_display]
    )

    submit_btn.click(
        submit_correction,
        inputs=[invoice_id_state, vendor, invoice_no, invoice_date,
               tax, total, debit_account, credit_account],
        outputs=[result]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)