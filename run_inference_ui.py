import streamlit as st
import subprocess
import os
import glob
import threading
import json
import http.client
import time
from queue import Queue, Empty

st.set_page_config(page_title="BitNet vs Ollama Comparison", layout="wide")

def get_models():
    return glob.glob("models/**/*.gguf", recursive=True)

def get_ollama_models():
    try:
        conn = http.client.HTTPConnection("localhost", 11434)
        conn.request("GET", "/api/tags")
        response = conn.getresponse()
        if response.status == 200:
            data = json.loads(response.read().decode())
            return [m['name'] for m in data.get('models', [])]
    except Exception:
        return []
    return []

def run_inference(model, prompt, n_predict, threads, temp):
    command = [
        "python3", "run_inference.py",
        "-m", model,
        "-p", prompt,
        "-n", str(n_predict),
        "-t", str(threads),
        "-temp", str(temp)
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        return process
    except Exception as e:
        st.error(f"Error running inference: {e}")
        return None

def run_ollama_inference(model, prompt, n_predict, temp, queue):
    try:
        conn = http.client.HTTPConnection("localhost", 11434)
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": True,
            "options": {"num_predict": n_predict, "temperature": temp}
        })
        conn.request("POST", "/api/generate", payload)
        response = conn.getresponse()
        for line in response:
            if line:
                chunk = json.loads(line.decode())
                response_text = chunk.get("response", "")
                if response_text: queue.put(response_text)
                if chunk.get("done"): break
    except Exception as e:
        queue.put(f"\nError: {e}")
    finally:
        queue.put(None)

st.title("🚀 BitNet vs Ollama Comparison")

# Sidebar for visibility toggles and shared settings
st.sidebar.header("Model Visibility")
show_a = st.sidebar.toggle("Show Model A (BitNet)", value=True)
show_b = st.sidebar.toggle("Show Model B (BitNet)", value=True)
show_o = st.sidebar.toggle("Show Model of Ollama", value=True)

st.sidebar.header("Global Settings")
threads_val = st.sidebar.number_input("Threads (BitNet)", min_value=1, max_value=32, value=6)
common_prompt = st.sidebar.text_area("Common Prompt", value="Classify if the text is a customer complaint. Text: Package arrived on time.")

models = get_models()
ollama_models = get_ollama_models() if show_o else []

active_slots = []
if show_a: active_slots.append("A")
if show_b: active_slots.append("B")
if show_o: active_slots.append("O")

if not active_slots:
    st.warning("Please enable at least one model in the sidebar.")
    st.stop()

cols = st.columns(len(active_slots))
config = {}

for i, slot in enumerate(active_slots):
    with cols[i]:
        if slot == "A":
            st.header("Model A (BitNet)")
            config['ma'] = st.selectbox("Select Model", models, key="sel_a")
            config['na'] = st.number_input("N-Predict", value=5, key="n_a")
            config['ta'] = st.slider("Temperature", 0.0, 1.0, 0.2, key="temp_a")
            if st.button("Run A"):
                with st.status("Running A...") as s:
                    p = run_inference(config['ma'], common_prompt, config['na'], threads_val, config['ta'])
                    out = st.empty()
                    txt = ""
                    for line in iter(p.stdout.readline, ""):
                        txt += line
                        out.code(txt)
                    p.wait()
                    s.update(label="Complete", state="complete")
        
        elif slot == "B":
            st.header("Model B (BitNet)")
            config['mb'] = st.selectbox("Select Model", models, index=min(1, len(models)-1), key="sel_b")
            config['nb'] = st.number_input("N-Predict", value=5, key="n_b")
            config['tb'] = st.slider("Temperature", 0.0, 1.0, 0.2, key="temp_b")
            if st.button("Run B"):
                with st.status("Running B...") as s:
                    p = run_inference(config['mb'], common_prompt, config['nb'], threads_val, config['tb'])
                    out = st.empty()
                    txt = ""
                    for line in iter(p.stdout.readline, ""):
                        txt += line
                        out.code(txt)
                    p.wait()
                    s.update(label="Complete", state="complete")

        elif slot == "O":
            st.header("Ollama")
            if not ollama_models:
                st.error("No Ollama models found.")
                config['mo'] = None
            else:
                config['mo'] = st.selectbox("Select Model", ollama_models, key="sel_o")
                config['no'] = st.number_input("N-Predict", value=5, key="n_o")
                config['to'] = st.slider("Temperature", 0.0, 1.0, 0.2, key="temp_o")
                if st.button("Run Ollama"):
                    with st.status("Running Ollama...") as s:
                        q = Queue()
                        threading.Thread(target=run_ollama_inference, args=(config['mo'], common_prompt, config['no'], config['to'], q), daemon=True).start()
                        out = st.empty()
                        txt = ""
                        while True:
                            it = q.get()
                            if it is None: break
                            txt += it
                            out.code(txt)
                        s.update(label="Complete", state="complete")

if len(active_slots) > 1:
    if st.button("🚀 Run All Active Models"):
        run_cols = st.columns(len(active_slots))
        placeholders = [c.empty() for c in run_cols]
        queues = []
        threads = []
        
        def enqueue_bitnet(out, q):
            for line in iter(out.readline, ''): q.put(line)
            out.close(); q.put(None)

        for slot in active_slots:
            q = Queue()
            queues.append(q)
            if slot == "A":
                p = run_inference(config['ma'], common_prompt, config['na'], threads_val, config['ta'])
                threads.append(threading.Thread(target=enqueue_bitnet, args=(p.stdout, q), daemon=True))
            elif slot == "B":
                p = run_inference(config['mb'], common_prompt, config['nb'], threads_val, config['tb'])
                threads.append(threading.Thread(target=enqueue_bitnet, args=(p.stdout, q), daemon=True))
            elif slot == "O":
                threads.append(threading.Thread(target=run_ollama_inference, args=(config['mo'], common_prompt, config['no'], config['to'], q), daemon=True))
        
        for t in threads: t.start()
        
        full_texts = [""] * len(active_slots)
        dones = [False] * len(active_slots)
        
        while not all(dones):
            updated = False
            for i in range(len(active_slots)):
                if not dones[i]:
                    try:
                        while True:
                            line = queues[i].get_nowait()
                            if line is None: dones[i] = True; break
                            full_texts[i] += line
                            updated = True
                    except Empty: pass
            if updated:
                for i in range(len(active_slots)):
                    placeholders[i].code(full_texts[i])
            time.sleep(0.05)
        st.success("Comparison Finished!")
