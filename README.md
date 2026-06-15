# Zig Sharded Inference Engine

**Production-ready distributed LLM inference engine built entirely in Zig** — bypassing Python runtime overhead to unlock maximum hardware efficiency.

---

## 🚀 Quick Start (Non-Technical)

### What Is This?

This is a **super-fast AI text generator** that runs on your computer or a cluster of computers. Instead of using slow Python-based tools (like most AI apps), it's built from scratch in **Zig** — a language that gives it direct control over your hardware for maximum speed.

Think of it like this:
- **Normal AI tools**: Driving a bus with a driver (Python) who makes lots of stops (memory allocations, garbage collection)
- **Zig Inference**: Driving a race car with direct control — no stops, full throttle

### What Can It Do?

| Feature | Benefit |
|---------|---------|
| **Chat with AI** | Generate text, answer questions, write stories |
| **Multi-GPU Support** | Use 2, 4, 8+ GPUs together for faster responses |
| **Multi-Node** | Connect multiple computers to run huge models |
| **No Python Required** | Lightweight — runs on bare metal, no heavy dependencies |
| **HTTP API** | Connect to any app (web, mobile, desktop) |

### How to Use It (3 Steps)

```bash
# 1. Download and run
./Zig_inference.sh all

# 2. Start the server
./Zig_inference.sh run /models/your-model.gguf 1 8080

# 3. Send a request
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'
```

**Expected output:**
```json
{"completion":"I'm doing well, thank for asking! How can I help you today?"}
```

### What You Need

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Linux (Ubuntu 22.04+) | Linux (Ubuntu 24.04) |
| **RAM** | 8 GB | 32 GB+ |
| **GPU** | None (CPU works) | NVIDIA RTX 3090/4090 or A100 |
| **Disk** | 10 GB | 100 GB+ (for models) |

### Downloading AI Models

You need an AI model file (`.gguf` format). Get one from:

1. **Hugging Face**: https://huggingface.co/models (search for "GGUF")
2. **Recommended models**:
   - `Llama 3.2 3B` — Fast, good for chat
   - `Mistral 7B` — Balanced speed/quality
   - `Qwen 14B` — High quality, needs more RAM

**Example download:**
```bash
curl -L "https://huggingface.co/llama-3.2-3b.gguf" -o /models/llama-3.2-3b.gguf
```

### Common Use Cases

#### 1. Chat Bot
```python
import requests

response = requests.post(
    "http://localhost:8080/generate",
    json={"prompt": "What is quantum computing?", "max_tokens": 200}
)
print(response.json()["completion"])
```

#### 2. Content Generation
```bash
curl -X POST http://localhost:8080/generate \
  -d '{"prompt": "Write a product description for a smartwatch", "max_tokens": 150}'
```

#### 3. Code Assistant
```bash
curl -X POST http://localhost:8080/generate \
  -d '{"prompt": "Write a Python function to sort a list", "max_tokens": 100}'
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `Zig not found` | Run `./Zig_inference.sh setup` to auto-install Zig |
| `Model not found` | Check path: `ls /models/your-model.gguf` |
| `CUDA not found` | Install NVIDIA CUDA Toolkit (optional — CPU works) |
| `Slow response` | Use smaller model (3B instead of 7B) or add GPU |

---

## 🛠️ Technical Documentation

### Architecture Overview
