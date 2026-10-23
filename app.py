from io import BytesIO
import zipfile
import uuid
from PIL import Image
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
    REGISTRY,
)
import streamlit as st
from transformers import BlipForConditionalGeneration, BlipProcessor

for name in [
    "images_processed_total",
    "app_active_requests",
    "image_inference_latency_seconds",
    "image_processing_duration_seconds",
]:
  if name in REGISTRY._names_to_collectors:
    try:
      REGISTRY.unregister(REGISTRY._names_to_collectors[name])
    except:
      pass

COUNTER = Counter(
    "images_processed_total",
    "Total images processed",
    ["mode", "status", "session_id"],
)
GAUGE = Gauge("app_active_requests", "Current active requests")
HISTOGRAM = Histogram(
    "image_inference_latency_seconds", "Inference time per image", ["mode"]
)
SUMMARY = Summary(
    "image_processing_duration_seconds",
    "Duration of image processing summary",
    ["mode"],
)

try:
  start_http_server(8000)
except:
  pass


@st.cache_resource
def get_model():
  return (
      BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
      BlipForConditionalGeneration.from_pretrained(
          "Salesforce/blip-image-captioning-base"
      ).to("cpu"),
  )


processor, model = get_model()

if "sid" not in st.session_state:
  st.session_state["sid"] = str(uuid.uuid4())
sid = st.session_state["sid"]

st.title("Image Captioning App")
mode = st.radio("Mode", ["Single", "Bulk"])

if mode == "Single":
  if file := st.file_uploader("Upload Image", type=["jpg", "png"]):
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)
    if st.button("Generate"):
      GAUGE.inc()
      try:
        with HISTOGRAM.labels(mode="Single").time(), SUMMARY.labels(
            mode="Single"
        ).time():
          inputs = processor(images=img, return_tensors="pt")
          outputs = model.generate(**inputs, max_new_tokens=50)
          caption = processor.decode(outputs[0], skip_special_tokens=True)
        COUNTER.labels(
            mode="Single", status="success", session_id=sid
        ).inc()
        st.success(caption)
      except Exception as e:
        COUNTER.labels(mode="Single", status="error", session_id=sid).inc()
        st.error(f"Failed: {e}")
      finally:
        GAUGE.dec()

else:
  if zip_file := st.file_uploader("Upload ZIP", type=["zip"]):
    if st.button("Process Batch"):
      GAUGE.inc()
      try:
        with zipfile.ZipFile(BytesIO(zip_file.read())) as z:
          for name in z.namelist():
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
              with z.open(name) as f:
                img = Image.open(f).convert("RGB")
                try:
                  with HISTOGRAM.labels(mode="Bulk").time(), SUMMARY.labels(
                      mode="Bulk"
                  ).time():
                    inputs = processor(images=img, return_tensors="pt")
                    outputs = model.generate(**inputs, max_new_tokens=50)
                    caption = processor.decode(
                        outputs[0], skip_special_tokens=True
                    )
                  COUNTER.labels(
                      mode="Bulk", status="success", session_id=sid
                  ).inc()
                  st.write(f"**{name}:** {caption}")
                except Exception as e:
                  COUNTER.labels(
                      mode="Bulk", status="error", session_id=sid
                  ).inc()
                  st.error(f"{name} failed: {e}")
      finally:
        GAUGE.dec()