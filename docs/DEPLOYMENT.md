# Deployment Guide

**Once you've trained a model with nanochat, you'll want to deploy it.**

---

## 🚀 Quick Start (Desktop)

For local testing and development, you can use the built-in web UI:

```bash
# Start the chat web server
python -m scripts.chat_web
```

Then open your browser to the URL shown (usually `http://localhost:8000`).

---

## 🌐 Desktop / Server Deployment

For production deployment on servers or desktop machines:

### Option 1: Ollama (Recommended)

```bash
# Create Ollama Modelfile
cat > Modelfile <<EOF
FROM ./checkpoint.pth
PARAMETER num_layer 26
PARAMETER num_head 16
PARAMETER num_embd 1024
LICENSE MIT
EOF

# Create model in Ollama
ollama create nanochat-model -f Modelfile

# Run inference
ollama run nanochat-model
```

### Option 2: vLLM (High Performance)

```bash
pip install vllm

python -m vllm.entrypoints.api_server \
    --model ./checkpoint.pth \
    --host 0.0.0.0 \
    --port 8000
```

---

## 🔌 Edge Deployment (Microcontrollers)

For deploying to resource-constrained devices like ESP32, STM32, or Raspberry Pi Pico:

### Q-Lite Gateway

**[Q-Lite](https://github.com/RalphBigBear/q-lite)** is an ultra-lightweight LLM gateway designed specifically for edge devices.

**Why Q-Lite?**
- **<1MB RAM** - Runs on ESP32, Raspberry Pi Zero
- **69KB binary** - Smaller than most LLM models
- **Ollama-compatible** - Drop-in replacement for Ollama API
- **Pure C** - Zero dependencies, runs everywhere

**Architecture**:
```
Edge Device (Q-Lite, <1MB RAM)
    ↓ HTTP
Desktop (Ollama, 128GB RAM)
    ↓
Response
```

### Example Workflow

```bash
# 1. Train your model with nanochat
bash runs/speedrun.sh

# 2. Convert checkpoint to Ollama format (TODO: add export script)
# 3. Serve model with Ollama on desktop
ollama serve

# 4. Deploy gateway to ESP32
cd q-lite/platforms/esp32
idf.py build
idf.py flash

# 5. Start Q-Lite gateway on ESP32
# (It connects to your desktop's Ollama via WiFi)
```

**Hardware Examples**:

| Device | RAM | Flash | Network | Q-Lite Binary |
|--------|-----|-------|---------|---------------|
| ESP32-S3 | 520KB | 4MB | WiFi | ~100KB |
| STM32F4 | 128KB | 512KB | Ethernet | ~80KB |
| Raspberry Pi Pico | 264KB | 2MB | WiFi (ESP8266) | ~60KB |
| Desktop | Plenty | N/A | Ethernet/WiFi | 69KB |

### Q-Lite Quick Start

```bash
# Clone Q-Lite
git clone https://github.com/RalphBigBear/q-lite.git
cd q-lite

# One-click demo
./examples/quickstart.sh
```

**NanoChat → Q-Lite Integration**:

1. **Train** with NanoChat (on desktop/HPC)
2. **Serve** with Ollama (on desktop/server)
3. **Deploy** with Q-Lite (to edge devices)

**Use Cases**:
- Home automation (ESP32 gateway + Ollama on NAS)
- IoT devices (Pico gateway + cloud LLM)
- Offline inference (Pico + local LLM)
- Teaching embedded AI (minimal hardware)

---

## 📊 Performance Comparison

| Deployment | Latency | Cost | Hardware |
|------------|---------|------|----------|
| Ollama Desktop | ~50ms | High ($1000 GPU) | Desktop |
| vLLM Server | ~20ms | Very High ($10K GPU) | Server |
| Q-Lite + Ollama | ~100ms | Low ($10 ESP32 + desktop) | Distributed |
| Q-Lite + Cloud | ~500ms | Low (data costs) | Edge |

**Trade-offs**:
- **Desktop**: Lowest latency, highest cost
- **Edge**: Higher latency, lowest cost, offline capable

---

## 🔧 Configuration

### Model Size Selection

Nanochat's `--depth` parameter controls model size:

| Depth | Parameters | RAM (inference) | Use Case |
|-------|-----------|------------------|----------|
| 12 | ~300M | ~1GB | Raspberry Pi, weak laptops |
| 20 | ~1B | ~3GB | Desktop, gaming PC |
| 26 (GPT-2) | ~1.6B | ~5GB | Server, powerful desktop |
| 30+ | ~3B+ | ~10GB+ | HPC, cloud |

**For edge deployment**:
- Use smaller models (d12-d16) for edge devices
- Run larger models (d20-d26) on desktop/server
- Q-Lite acts as gateway between edge and model

---

## 📖 References

- **Q-Lite GitHub**: https://github.com/RalphBigBear/q-lite
- **NanoChat Training**: https://github.com/karpathy/nanochat
- **Ollama Docs**: https://ollama.com/docs
- **vLLM Docs**: https://docs.vllm.ai

---

**Inspired by**: [OpenClaw Discussion #14132](https://github.com/openclaw/openclaw/discussions/14132)

**Special thanks** to @karpathy for proving that minimalism beats feature bloat.
