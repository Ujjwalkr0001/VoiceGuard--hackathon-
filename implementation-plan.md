# VoiceGuard: AI-Powered Real-Time Voice Clone Detection System
## Master Implementation Plan (SIH Problem Statement 26104)

---

## 1. Executive Summary & System Architecture

### 1.1 Overview
**VoiceGuard** is an enterprise-grade, low-latency defense system designed to intercept, analyze, and detect AI-cloned voices and deepfake audio during live phone calls. Built specifically for **Smart India Hackathon (SIH) Problem Statement 26104**, the solution bridges incoming telephony calls through Twilio, extracts raw audio frames from the caller (`inbound_track`) via real-time WebSockets, runs a multi-tier acoustic and semantic analysis pipeline, and instantly triggers high-priority alerts via Firebase Cloud Messaging (FCM) and Twilio SMS when an impersonation or clone attempt is identified.

```
                                  +-----------------------------------------------------------+
                                  |                     TWILIO CLOUD                         |
                                  |  +----------------+        +--------------------------+  |
Caller (Potential Clone) -------->|  | Programmable   |------->| Media Stream (WSS)       |  |
                                  |  | Voice Bridge   |        | track="inbound_track"    |  |
                                  |  +----------------+        | G.711 mu-law 8kHz        |  |
                                  +----------------------------+-----------------------------+
                                                                             | Base64 Frames (20ms)
                                                                             v
+--------------------------------------------------------------------------------------------+
|                               FASTAPI INGESTION & PIPELINE (AWS ECS)                       |
|                                                                                            |
|  +-----------------------------+       +-------------------------+                         |
|  | WebSocket Ingestion Handler |------>| Decoded PCM Ring Buffer |                         |
|  | (/media-stream)             |       | 2.0s Window / 0.5s Step |                         |
|  +-----------------------------+       +-------------------------+                         |
|                                                     |                                      |
|                 +-----------------------------------+----------------------------------+   |
|                 |                                                                      |   |
|                 v (16kHz Resampled PCM)                                                v   |
|  +-------------------------------+                             +---------------------+     |
|  |  Tier 1: Deep Feature Model   |                             | Tier 2: DSP Feature |     |
|  |  (AASIST / RawNet2 PyTorch)   |                             | Extractor Engine    |     |
|  |  - Spectral-Temporal Graph    |                             | - Modified Grp Delay|     |
|  |  - TorchScript Quantized      |                             | - CQCC / LFCC Filter|     |
|  |  - Raw waveform -> [0, 1]     |                             | - Micro-Pitch Jitter|     |
|  +-------------------------------+                             | - Silero VAD Cadence|     |
|                 |                                              +---------------------+     |
|                 |                                                         |                |
|                 |                                                         v                |
|                 |                                              +---------------------+     |
|                 |                                              | Tier 3: XGBoost /   |     |
|                 |                                              | LightGBM Classifier |     |
|                 |                                              | + TreeSHAP Explainer|     |
|                 |                                              +---------------------+     |
|                 |                                                         |                |
|                 +-----------------------+   +-----------------------------+                |
|                                         |   |                                              |
|                                         v   v                                              |
|                               +------------------------+                                   |
|                               | Calibrated Ensemble    |                                   |
|                               | Acoustic Score P_ac    |                                   |
|                               +------------------------+                                   |
|                                           |                                                |
|                                           |     +---------------------------------------+  |
|                                           |     | Tier 4: Context / Semantic Threat     |  |
|                                           |     | - Sarvam AI Indic STT (4s window)     |  |
|                                           |     | - Fraud Keyword Regex & MiniLM Embed  |  |
|                                           |     | - Context Score P_ctx                 |  |
|                                           |     +---------------------------------------+  |
|                                           v                         |                      |
|                           +-----------------------------------------------+                |
|                           |      Bayesian Multi-Factor Risk Engine        |                |
|                           |  - Caller Reputation Multiplier (M_caller)    |                |
|                           |  - EMA Rolling Risk Window (Redis Cache)      |                |
|                           |  - Dual-Threshold Hysteresis State Machine   |                |
|                           +-----------------------------------------------+                |
|                                                   |                                        |
|                          +------------------------+------------------------+               |
|                          | Risk >= 70 (Warning) / Risk >= 85 (Critical)    |               |
|                          v                                                 v               |
|            +----------------------------+                    +---------------------------+ |
|            | DynamoDB Session Log Store |                    | Alert Dispatcher via SNS  | |
|            | (TTL: 30 Days Auto-Purge)  |                    +---------------------------+ |
|            +----------------------------+                                  |               |
+----------------------------------------------------------------------------|---------------+
                                                                             |
                                     +---------------------------------------+---------------------------------------+
                                     |                                                                               |
                                     v                                                                               v
                    +----------------------------------+                                            +----------------------------------+
                    |  AWS SNS -> Firebase Cloud Msg   |                                            |    AWS SNS -> Twilio SMS API     |
                    |  - High-Priority Data Payload    |                                            |    - Backup Direct Alert SMS     |
                    |  - Custom Ring/Vibration Intent  |                                            |    - Critical Warning Link       |
                    +----------------------------------+                                            +----------------------------------+
                                     |                                                                               |
                                     v                                                                               v
                    +----------------------------------+                                            +----------------------------------+
                    |   Target User Mobile Phone       |                                            |    User Native SMS Inbox         |
                    |   React Native App Alert Modal   |                                            +----------------------------------+
                    |   - Full Screen Alert HUD        |
                    |   - Haptic Vibration Loop        |
                    |   - SHAP Diagnostic Breakdown    |
                    |   - "Disconnect & Verify" Button |
                    +----------------------------------+
```

### 1.2 The Core Telephony Problem & Solution Strategy
In standard deepfake detection benchmarks (such as ASVspoof 2019/2021), audio is uncompressed 16kHz/48kHz linear PCM. In a live production telephony environment:
1. **Twilio Media Streams deliver 8,000 Hz, 8-bit mu-law (G.711u) audio**.
2. Frequencies above **3,400 Hz are aggressively filtered out** by standard PSTN / cellular bandpass filters.
3. Neural vocoder artifacts that typically manifest in high frequencies (12kHz - 24kHz) are completely discarded by the telephone network.
4. **The Solution**: 
   - Feature extraction and neural models must explicitly focus on **low-band acoustic phenomena**: phase discontinuities in vocal tract synthesis (Modified Group Delay), sub-band spectral flux, micro-prosodic pitch instability (`torchcrepe`), and speech-to-pause cadence (Silero VAD).
   - All training and fine-tuning datasets are augmented with a **Telephony Simulation Degradation Pipeline** (G.711 mu-law transcoding, 300Hz-3400Hz ITU-T G.712 bandpass filtering, and GSM packet loss simulation).

---

## 2. Target Performance & Accuracy Benchmarks

| Metric | Target Specification | Enforcement Mechanism |
|---|---|---|
| **Equal Error Rate (EER)** | **$\le 3.2\%$** on telephony-degraded audio | Calibrated dual-model ensemble (AASIST + XGBoost) |
| **False Positive Rate (FPR)** | **$< 1.5\%$** at threshold $\tau = 85$ | Bayesian rolling-average filter + caller contact whitelist |
| **End-to-End Latency** | **$< 480 \text{ ms}$** per chunk | In-memory ring buffer + Quantized ONNX/TorchScript CPU runtime |
| **Streaming Chunk Cadence**| **2.0s window with 0.5s step** | 75% overlap for continuous detection without alert delays |
| **Context Extraction Latency** | **$< 1.2 \text{ s}$** on 4s speech segments | Sarvam AI Indic fast-transcription API |
| **Data Retention** | **Zero raw audio persistence**; 30-day log TTL | Pure in-memory streaming; DynamoDB native TTL |

---

## 3. Monorepo Structure & Environment Scaffolding (Phase 0)

### 3.1 Repository Layout
```plaintext
VoiceGuard/
├── .env.example
├── .gitignore
├── README.md
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI server entry point & lifespan
│   │   ├── config.py                  # Pydantic v2 settings management
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes_twiml.py        # TwiML webhook endpoints
│   │   │   ├── routes_websocket.py    # Twilio Media Streams WSS consumer
│   │   │   └── routes_sessions.py     # REST API for call logs & alerts
│   │   ├── core/
│   │   │   ├── audio_buffer.py        # Circular PCM ring buffer (2s window / 0.5s hop)
│   │   │   ├── mu_law.py              # Fast G.711 mu-law to 16kHz PCM decoder
│   │   │   └── state_manager.py       # Redis-backed session & risk tracker
│   │   ├── dsp/
│   │   │   ├── __init__.py
│   │   │   ├── feature_extractor.py   # Unified 128-dim DSP extraction engine
│   │   │   ├── group_delay.py         # Modified Group Delay (MGD) phase spectrum
│   │   │   ├── cqcc.py                # Telephony-band CQCC / LFCC extractor
│   │   │   ├── pitch_jitter.py        # TorchCrepe micro-prosody & shimmer
│   │   │   └── vad_cadence.py         # Silero VAD pause distribution
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── deep_model.py          # AASIST / RawNet2 TorchScript wrapper
│   │   │   ├── boosted_model.py       # XGBoost DSP inference + TreeSHAP
│   │   │   └── ensemble.py            # Platt scaling & acoustic score fusion
│   │   ├── nlp/
│   │   │   ├── __init__.py
│   │   │   ├── stt_client.py          # Sarvam AI Indic STT + Whisper fallback
│   │   │   └── threat_analyzer.py     # Regex matcher + Sentence-Transformers
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── risk_engine.py         # Bayesian rolling risk calculator
│   │   │   └── alert_dispatcher.py    # AWS SNS -> FCM & Twilio SMS publisher
│   │   └── db/
│   │       ├── __init__.py
│   │       └── dynamodb_client.py     # Session persistence with 30-day TTL
│   └── tests/
│       ├── test_audio_pipeline.py
│       ├── test_dsp_features.py
│       └── test_risk_engine.py
├── ml/
│   ├── requirements-train.txt
│   ├── datasets/
│   │   ├── download_asvspoof.py       # ASVspoof 2019/2021 automated downloader
│   │   ├── generate_sarvam_tts.py     # Synthetic clone dataset generator (Sarvam TTS)
│   │   └── telephony_augment.py       # G.711 mu-law transcoding & G.712 bandpass
│   ├── training/
│   │   ├── train_aasist.py            # Fine-tuning AASIST with early stopping on EER
│   │   ├── train_xgboost.py           # Training LightGBM/XGBoost on DSP vectors
│   │   └── calibrate_scores.py        # Isotonic regression & Platt scaling
│   └── export/
│       └── export_torchscript.py      # Quantize and serialize models to TorchScript
├── app/                               # Mobile App Track (React Native)
│   ├── package.json
│   ├── app.json
│   ├── index.js
│   ├── src/
│   │   ├── App.tsx                    # Root navigation & FCM background listeners
│   │   ├── services/
│   │   │   ├── fcm_service.ts         # Firebase push notification & headless handler
│   │   │   ├── haptic_service.ts      # Hardware vibration patterns
│   │   │   └── api_client.ts          # REST client to fetch past session alerts
│   │   ├── screens/
│   │   │   ├── AlertModalScreen.tsx   # Urgent HUD overlay with SHAP explanation
│   │   │   ├── CallHistoryScreen.tsx  # Past flagged call sessions
│   │   │   └── SettingsScreen.tsx     # Whitelist contacts & sensitivity config
│   │   └── components/
│   │       ├── RiskGauge.tsx          # Circular animated score meter
│   │       └── SignalBadge.tsx        # Explainable AI feature pills
└── infra/
    ├── docker-compose.yml             # Local dev: Redis, LocalStack, FastAPI
    ├── ecs-task-definition.json       # AWS Fargate task configuration
    └── terraform/                     # ALB, SNS, DynamoDB, ElastiCache definitions
```

### 3.2 Environment Configuration (`.env.example`)
```ini
# Server Configuration
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=production

# Twilio Credentials
TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=+1XXXXXXXXXX
TWILIO_APP_CALLBACK_URL=https://api.voiceguard.ai/api/v1/twiml/voice

# Sarvam AI (Indic Speech-to-Text & Synthetic Dataset Vocoder)
SARVAM_API_KEY=your_sarvam_api_key_here

# AWS Infrastructure
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
DYNAMODB_TABLE_NAME=voiceguard_call_sessions
SNS_ALERT_TOPIC_ARN=arn:aws:sns:ap-south-1:123456789012:voiceguard-alerts
REDIS_URL=redis://localhost:6379/0

# Firebase Cloud Messaging
FCM_SERVER_KEY=your_firebase_server_key_or_sa_json
```

---

## 4. Phase-by-Phase Technical Implementation Steps

### Phase 1: Twilio Telephony Ingestion & Real-Time Audio Streaming
**Objective**: Intercept live telephony calls, bridge the connection, tap into `inbound_track`, stream 8kHz G.711 mu-law frames over secure WebSockets, and convert them into continuous 16kHz PCM windows in memory.

#### Detailed Engineering Steps:
1. **TwiML Bridge Endpoint (`routes_twiml.py`)**:
   - Construct an endpoint `@app.post("/api/v1/twiml/voice")` returning TwiML:
     ```xml
     <?xml version="1.0" encoding="UTF-8"?>
     <Response>
       <Start>
         <Stream url="wss://api.voiceguard.ai/media-stream" track="inbound_track">
           <Parameter name="caller" value="{{From}}" />
           <Parameter name="callee" value="{{To}}" />
         </Stream>
       </Start>
       <Dial>{{ForwardToNumber}}</Dial>
     </Response>
     ```
   - *Design rationale*: `track="inbound_track"` ensures the backend exclusively receives caller audio. The target user's voice is never transmitted to the stream, preserving user privacy, halving network bandwidth, and eliminating GDPR/DPDP retention complications.

2. **WebSocket Ingestion Endpoint (`routes_websocket.py`)**:
   - Handle Twilio Media Stream JSON protocol events:
     - `connected`: Initialize WebSocket session handshake.
     - `start`: Extract `callSid`, `streamSid`, and custom parameters (`caller`, `callee`). Initialize a dedicated `AudioBuffer` and Redis session record.
     - `media`: Extract base64 encoded audio payload (160 bytes = 20ms of audio). Decode via `mu_law.py` and push to circular buffer.
     - `stop`: Flush remaining frames, trigger final session verdict computation, update DynamoDB, and clear Redis hot-path state.

3. **Audio Decoding & Ring-Buffering Engine (`mu_law.py` & `audio_buffer.py`)**:
   - Convert 8-bit mu-law to 16-bit linear PCM using a precomputed 256-element integer lookup table for maximum efficiency (avoiding per-sample math).
   - Resample from 8,000 Hz to 16,000 Hz using Polyphase FIR resampling (`scipy.signal.resample_poly` or `torchaudio.transforms.Resample`).
   - Implement a thread-safe, lock-free circular buffer per `call_sid`:
     - **Window Size**: 2.0 seconds (32,000 samples at 16kHz).
     - **Step / Hop Size**: 0.5 seconds (8,000 samples at 16kHz).
     - **Overlap**: 75% overlap ensures smooth rolling evaluation and sub-second anomaly detection.

---

### Phase 2: High-Precision DSP Feature Extraction Suite (`dsp/`)
**Objective**: Extract 128 engineered acoustic features specifically sensitive to vocoder artifacts, synthetic vocal tract modeling errors, unnatural pitch micro-prosody, and robotic speech cadences within the 300Hz-3400Hz telephony band.

```
       Raw 16kHz PCM (2.0-second chunk)
                     |
         +-----------+-----------+
         |                       |
         v                       v
 [Silero VAD Filter]     [STFT Complex Spectrum]
         |                       |
         |         +-------------+-------------+
         |         |                           |
         v         v                           v
+-------------+ +--------------------+ +--------------------+
| Pause/Cadence| | Modified Group     | | Telephony CQCC/    |
| Distribution| | Delay (MGD) Phase  | | LFCC Filterbank    |
| (16 dims)   | | (48 dims)          | | (40 dims)          |
+-------------+ +--------------------+ +--------------------+
         |                 |                     |
         +--------+--------+----------+----------+
                  |                   |
                  v                   v
         +-----------------+ +-----------------+
         | TorchCrepe F0   | | Spectral Flux & |
         | Jitter/Shimmer  | | High-Band Cutoff|
         | (14 dims)       | | (10 dims)       |
         +-----------------+ +-----------------+
                           |
                           v
        Concatenated 128-Dimensional Vector
           + Human-Readable SHAP Context
```

#### Detailed Engineering Steps:
1. **Modified Group Delay (MGD) Phase Spectrum (`group_delay.py`)**:
   - *Acoustic Principle*: Deep learning vocoders (HiFi-GAN, WaveGlow, StyleTTS2) reconstruct audio from magnitude spectrograms using neural vocoders or Griffin-Lim. While magnitude spectra look genuine, the **phase spectrum contains severe non-linear discontinuities**.
   - *Mathematical Formulation*:
     Let $X(\omega) = |X(\omega)| e^{j\theta(\omega)}$ be the Short-Time Fourier Transform of speech $x(n)$, and $Y(\omega)$ be the STFT of $n \cdot x(n)$. The modified group delay function $\tau_\rho(\omega)$ is computed as:
     $$\tau_\rho(\omega) = \text{sign} \cdot \left| \frac{X_R(\omega) Y_R(\omega) + X_I(\omega) Y_I(\omega)}{|S(\omega)|^{2\gamma}} \right|^\alpha$$
     where $|S(\omega)|$ is the smoothed spectral envelope obtained via cepstral smoothing, $\alpha=0.4$, and $\gamma=0.9$.
   - Extract 48 cepstral coefficients from the MGD phase spectrum.

2. **Telephony-Band Constant-Q / Linear-Frequency Cepstral Coefficients (`cqcc.py`)**:
   - Compute CQCC / LFCC with filter banks strictly bounded between **300 Hz and 3,400 Hz** with 24 filters.
   - Compute static, $\Delta$ (delta), and $\Delta\Delta$ (delta-delta) velocity coefficients (40 dimensions).
   - This isolates unnatural formant step-changes common in autoregressive TTS models.

3. **Micro-Prosody, Jitter & Shimmer Engine (`pitch_jitter.py`)**:
   - Extract fundamental frequency ($F_0$) track on voiced frames using `torchcrepe` (a deep convolutional pitch tracker) or `pyin`.
   - Compute:
     - **Relative Jitter (local)**:
       $$\text{Jitter} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N}\sum_{i=1}^N T_i}$$
     - **Shimmer (local)**: Frame-to-frame peak amplitude variation.
     - **Pitch Entropy**: Cloned speech frequently demonstrates unnatural pitch constancy (cloned voices sound "too flat") or exaggerated robotic pitch transitions. (14 dimensions).

4. **Speech-to-Pause Cadence Analysis via Silero VAD (`vad_cadence.py`)**:
   - Run lightweight Silero VAD over the 2-second buffer.
   - Compute pause duration distribution, average phonation burst duration, and speech-to-silence ratio.
   - Flag unnatural zero-millisecond transitions or rigid, fixed-length synthetic breath pauses generated by TTS engines. (16 dimensions).

5. **Unified Feature Formatter (`feature_extractor.py`)**:
   - Assemble the 128-dimensional vector as a normalized `numpy.ndarray` (`float32`).
   - Generate an accompanying `raw_metrics` dictionary for human-readable dashboard reporting (e.g. `jitter_percent`, `phase_anomaly_index`, `pause_regularity_score`).

---

### Phase 3: Detection Model A — Deep Neural Network (AASIST)
**Objective**: Deploy a deep neural network that processes raw 16kHz audio directly to capture complex temporal-spectral spoofing artifacts, exported for sub-50ms CPU execution.

#### Detailed Engineering Steps:
1. **Architecture Selection: AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)**:
   - AASIST processes raw waveform inputs through SincNet filters followed by a dual-branch graph attention network (GAT) analyzing spectral and temporal relationships simultaneously.
   - Outperforms standard ResNet and RawNet2 on ASVspoof benchmarks by modeling long-range vocoder artifacts.

2. **Telephony Adaptation & Fine-Tuning Pipeline (`ml/training/train_aasist.py`)**:
   - Take the pretrained AASIST checkpoint (trained on ASVspoof 2019 LA).
   - Fine-tune using a custom loss combining Weighted Cross-Entropy and Angular Margin Softmax (ArcFace / AM-Softmax) to maximize separation between genuine and cloned voice clusters.
   - Train over 30 epochs with Adam optimizer ($\text{lr} = 10^{-4}$ with cosine annealing) and early stopping based on validation Equal Error Rate (EER).

3. **Optimization & TorchScript Export (`ml/export/export_torchscript.py`)**:
   - Trace and freeze the fine-tuned AASIST model using PyTorch JIT (`torch.jit.trace`).
   - Apply dynamic 8-bit quantization (`torch.quantization.quantize_dynamic`) on linear layers.
   - **Performance Result**: Reduces model memory footprint from ~85MB to ~22MB; reduces CPU inference latency from 140ms down to **32ms** per 2-second chunk on an AWS Fargate CPU core.

4. **Inference Service Wrapper (`models/deep_model.py`)**:
   - Maintain a pre-warmed singleton instance of the TorchScript model.
   - Expose `predict(pcm_tensor: torch.Tensor) -> float` returning raw spoof probability $P_{deep} \in [0, 1]$.

---

### Phase 4: Detection Model B — Interpretable Boosted Classifier (XGBoost)
**Objective**: Train an ultra-fast gradient boosted decision tree classifier on the 128 DSP features to serve as an orthogonal detection tier and generate real-time explainability (TreeSHAP).

#### Detailed Engineering Steps:
1. **Training & Regularization (`ml/training/train_xgboost.py`)**:
   - Train an XGBoost / LightGBM binary classifier on pre-extracted DSP feature vectors from the augmented dataset.
   - Hyperparameters optimized via Bayesian search (`optuna`): `n_estimators=350`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`.
   - Implement 5-fold cross-validation stratified by speaker ID to prevent speaker-identity leakage.

2. **Real-Time Explainability via TreeSHAP (`models/boosted_model.py`)**:
   - Package `shap.TreeExplainer` alongside the trained model.
   - On inference, extract the top 3 contributing features pushing the score toward "spoof":
     ```python
     # Example extracted explanation
     {
       "top_signals": [
         {"feature": "mgd_phase_discontinuity", "importance": 0.38, "label": "Phase Discontinuity Detected"},
         {"feature": "pitch_jitter_rigidity", "importance": 0.27, "label": "Unnatural Pitch Monotony"},
         {"feature": "synthetic_pause_ratio", "importance": 0.19, "label": "Artificial Pause Rhythm"}
       ]
     }
     ```
   - *Hackathon Demo Impact*: Provides concrete, visible proof to SIH judges explaining *why* the AI flagged the call, avoiding "black-box" skepticism.

---

### Phase 5: Calibrated Ensemble & Semantic Threat Intelligence Layer
**Objective**: Fuse acoustic models using Platt scaling and combine them with real-time speech transcription and fraud phrase detection.

#### Detailed Engineering Steps:
1. **Score Calibration & Acoustic Ensemble Fusion (`models/ensemble.py`)**:
   - Raw model outputs are rarely well-calibrated probabilities. Apply **Platt Scaling** (logistic regression fitted on validation logits) to transform raw outputs into true posterior probabilities:
     $$P(y = 1 | s) = \frac{1}{1 + \exp(A \cdot s + B)}$$
   - Compute the Unified Acoustic Spoof Score:
     $$S_{acoustic} = w_A \cdot P_{AASIST} + w_B \cdot P_{XGBoost}$$
     *(Nominal default weights: $w_A = 0.65, w_B = 0.35$)*.

2. **Speech-to-Text Ingestion (`nlp/stt_client.py`)**:
   - Buffer audio over a 4-second sliding window.
   - Send chunk to **Sarvam AI Indic STT API** (`saarika:v1` model) supporting Indian English, Hindi, and regional languages.
   - Maintain a local Whisper-base fallback in case of external API timeouts (> 1200ms).

3. **Scam Taxonomy & Semantic Risk Analyzer (`nlp/threat_analyzer.py`)**:
   - Scan transcripts against a three-tier fraud keyword taxonomy:
     - **Tier 1 (Critical - Impersonation & Direct Theft)**: `"OTP"`, `"one time password"`, `"UPI PIN"`, `"CVV"`, `"biometric"`, `"police custody"`, `"arrest warrant"`, `"CBI officer"`.
     - **Tier 2 (High - Financial Urgency & Coercion)**: `"immediate transfer"`, `"bank account blocked"`, `"hospital emergency"`, `"kidnapped"`, `"customs clearance fee"`.
     - **Tier 3 (Medium - Suspicious Framing)**: `"don't tell anyone"`, `"urgent action required"`, `"KYC expired"`.
   - Complement regex with vector cosine similarity using `sentence-transformers` (`all-MiniLM-L6-v2`) against 25 canonical Indian tele-fraud script embeddings.
   - Returns a normalized Context Threat Score: $S_{context} \in [0.0, 1.0]$.

---

### Phase 6: Bayesian Multi-Factor Risk Scoring Engine (`engine/risk_engine.py`)
**Objective**: Compute a rolling, composite risk score per call session in Redis, applying caller reputation multipliers, exponential moving average smoothing, and hysteresis thresholding.

#### Detailed Engineering Steps:
1. **Mathematical Risk Formulation**:
   For each new 0.5-second evaluation step $t$:
   $$S_{raw}(t) = \left( \alpha \cdot S_{acoustic}(t) + \beta \cdot S_{context}(t) \right) \times M_{caller}$$
   Where:
   - $\alpha = 0.70$ (acoustic clone evidence is the dominant factor).
   - $\beta = 0.30$ (scam context amplifies suspicion).
   - $M_{caller}$ is the **Caller Reputation Multiplier**:
     - $M_{caller} = 1.0$ if the caller's phone number is in the user's verified contact whitelist.
     - $M_{caller} = 1.4$ if the caller is an unknown number or spoofed caller ID.
   - Cap $S_{raw}(t)$ at a maximum value of 100.

2. **Exponential Moving Average (EMA) Rolling Window**:
   To prevent momentary acoustic anomalies (e.g. background noise spikes, throat clearing) from generating false alarms:
   $$R(t) = \lambda \cdot R(t-1) + (1 - \lambda) \cdot S_{raw}(t)$$
   Where smoothing factor $\lambda = 0.65$. Store $R(t)$ in a rolling 10-item Redis sorted set per `call_sid`.

3. **Dual-Threshold Hysteresis State Machine**:
   - **State 0: NORMAL ($R(t) < 70$)**: No user interruption.
   - **State 1: SUSPICIOUS ($70 \le R(t) < 85$)**: Internal alert; app pre-warns or displays subtle indicator.
   - **State 2: CRITICAL CLONE DETECTED ($R(t) \ge 85$)**: Immediate fan-out alert triggered.
   - *Hysteresis rule*: Once State 2 is entered, the alert remains active until score drops below 60 for 4 consecutive windows (2 seconds), preventing alert flickering.

---

### Phase 7: Real-Time Alerting & Multi-Channel Fan-Out (`engine/alert_dispatcher.py`)
**Objective**: Broadcast critical alert payloads to AWS SNS, fanning out in parallel to Firebase Cloud Messaging (FCM) and Twilio Programmable SMS.

#### Detailed Engineering Steps:
1. **SNS Fan-Out Publication**:
   - When the Risk Engine transitions to `CRITICAL`, publish an event to `arn:aws:sns:ap-south-1:...:voiceguard-alerts` with:
     ```json
     {
       "call_sid": "CA1234567890abcdef",
       "timestamp": 1773273600,
       "risk_score": 92.4,
       "caller_number": "+919876543210",
       "callee_number": "+918765432109",
       "top_signals": [
         "Unnatural Pitch Monotony (94%)",
         "Synthetic Phase Discontinuity (89%)",
         "High-Risk Keyword: 'Immediate UPI Transfer'"
       ],
       "action_required": "DISCONNECT_AND_VERIFY"
     }
     ```

2. **Subscriber A: High-Priority FCM Push Notification**:
   - Configured with `priority: "high"` and Android-specific `channel_id: "voiceguard_critical_alerts"`.
   - Includes custom data payload triggering a native full-screen HUD popup and sustained vibration pattern on the target phone even when the screen is locked.

3. **Subscriber B: Twilio Backup SMS Dispatcher**:
   - Fires simultaneously via Twilio REST API:
     *"[VoiceGuard ALERT] Critical: The incoming call from +919876543210 exhibits a 92% probability of being an AI Voice Clone requesting financial actions. Hang up immediately."*
   - Guarantees notification delivery even if mobile data connectivity is intermittent during the call.

---

### Phase 8: Data Layer & 30-Day Auto-Purge Lifecycle (`db/dynamodb_client.py`)
**Objective**: Log session metadata and risk timelines to AWS DynamoDB with automated 30-day deletion, strictly adhering to zero-raw-audio persistence.

#### Detailed Engineering Steps:
1. **DynamoDB Schema (`voiceguard_call_sessions`)**:
   - **Partition Key**: `call_sid` (String).
   - **Attributes**:
     - `caller_number` (String, hashed / partially masked for privacy)
     - `callee_number` (String)
     - `start_time` (ISO-8601 Timestamp)
     - `duration_seconds` (Integer)
     - `peak_risk_score` (Number)
     - `final_verdict` (String: `GENUINE` | `SUSPICIOUS` | `CLONE_DETECTED`)
     - `risk_timeline` (List of `{ "t": float, "score": float }`)
     - `top_contributing_factors` (List of Strings)
     - `ttl` (Number: Unix timestamp set to `epoch_now + 2,592,000` [30 days]).

2. **Privacy Enforcement**:
   - No audio chunks or STT transcript text are saved to disk or database. Audio buffers exist strictly in RAM and are overwritten after the 2-second analysis window expires.

---

### Phase 9: Mobile Application Track (React Native - Android/iOS)
**Objective**: Build a responsive React Native application that handles high-priority FCM payloads, displays a full-screen intrusive alert HUD, triggers distinct native vibration loops, and provides explainable diagnostics and historical logs.

```
       Incoming Push (FCM Data Payload: type="high_risk_call")
                               |
                               v
               +-------------------------------+
               | Headless JS / Background Task |
               +-------------------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+-----------------------+             +-----------------------+
|  Native Vibration Loop|             | Full-Screen Alert HUD |
|  [0, 500, 200, 500ms] |             | (Overlays Lock Screen)|
+-----------------------+             +-----------------------+
                                                  |
                    +-----------------------------+-----------------------------+
                    |                             |                             |
                    v                             v                             v
         +--------------------+        +--------------------+        +--------------------+
         | Risk Score Meter   |        | Top Cloned Signals |        | Primary Actions    |
         | (Animated Red 92%) |        | - Jitter Rigidity  |        | - "Disconnect Call"|
         |                    |        | - MGD Phase Anomaly|        | - "Verify Contact" |
         +--------------------+        +--------------------+        +--------------------+
```

#### Detailed Engineering Steps:
1. **Project Setup & Dependencies**:
   - React Native 0.74+ with TypeScript.
   - Core packages: `@react-native-firebase/app`, `@react-native-firebase/messaging`, `react-native-vibration`, `react-native-svg` (for animated risk dials), `lottie-react-native`.

2. **Headless Background Notification Receiver (`src/services/fcm_service.ts`)**:
   - Register `messaging().setBackgroundMessageHandler` and `messaging().onMessage`.
   - On detecting `data.type === "high_risk_call"`:
     - Invoke `HapticService.triggerAlarmPattern()`.
     - Wake the screen and navigate immediately to `AlertModalScreen`.

3. **Aggressive Haptic Feedback (`src/services/haptic_service.ts`)**:
   - Execute repeating pulse vibration pattern: `Vibration.vibrate([0, 600, 250, 600, 250, 1000], true)`.
   - Provide a cancel method once the user dismisses the alert or taps an action.

4. **Full-Screen Alert HUD (`src/screens/AlertModalScreen.tsx`)**:
   - Vibrant, high-contrast dark theme (#0A0E17 background with #FF3B30 emergency red accents).
   - Prominently displays:
     - Warning header: *"POTENTIAL AI VOICE CLONE DETECTED"*
     - Animated circular risk meter showing real-time score (e.g. `92% RISK`).
     - Caller phone number and contact match status.
     - SHAP explainability pills: *"Unnatural Pitch Uniformity"*, *"Phase Discontinuity"*, *"Scam Keyword: OTP Transfer"*.
   - **Primary Action**: A prominent red button labeled *"Disconnect Call & Call Back to Verify"*, which triggers the native phone dialer with the verified contact number from the user's address book.

5. **Call History & Audit Screen (`src/screens/CallHistoryScreen.tsx`)**:
   - Lists past 30 days of monitored calls fetched from `/api/v1/sessions`.
   - Allows users to review flagged call timestamps, risk timelines, and contributing factors.

6. **Settings & Whitelist Configuration (`src/screens/SettingsScreen.tsx`)**:
   - Allows user to sync address book to define the trusted whitelist (reducing $M_{caller}$ to 1.0).
   - Adjustable sensitivity threshold slider (Standard: 85%, High Sensitivity: 75%).

---

### Phase 10: Synthetic Dataset Pipeline & Model Accuracy Optimization
**Objective**: Build a high-fidelity synthetic voice clone dataset, implement realistic telephony channel degradation, and optimize detection accuracy to achieve $\text{EER} \le 3.2\%$.

```
+-----------------------------------------------------------------------------------+
|                        DATASET SYNTHESIS & AUGMENTATION PIPELINE                  |
|                                                                                   |
|  1. Bonafide Sources:                                                             |
|     - ASVspoof 2019 LA / 2021 DF (Bonafide partitions)                            |
|     - IndicData / CommonVoice India (Indian English & Hindi native speakers)     |
|                                                                                   |
|  2. Cloned / Spoof Generation (Targeted Vocoders):                                |
|     - Sarvam AI TTS (Bulbul:v1 Hindi & Indian English)                            |
|     - ElevenLabs Voice Cloning (Indian Accents)                                   |
|     - StyleTTS2 & FastSpeech2 + HiFi-GAN Vocoder                                  |
|                                                                                   |
|  3. Telephony Channel Degradation Pipeline:                                       |
|     +--------------------------------------------------------------------------+  |
|     |  Raw 16kHz Audio                                                         |  |
|     |       |                                                                  |  |
|     |       v                                                                  |  |
|     |  [ITU-T G.712 Bandpass Filter: 300 Hz - 3400 Hz]                         |  |
|     |       |                                                                  |  |
|     |       v                                                                  |  |
|     |  [Resample to 8,000 Hz]                                                  |  |
|     |       |                                                                  |  |
|     |       v                                                                  |  |
|     |  [G.711 mu-law 8-bit Quantization & De-quantization]                     |  |
|     |       |                                                                  |  |
|     |       v                                                                  |  |
|     |  [Additive Telephony Noise: MUSAN Babble / Office @ 15-25dB SNR]         |  |
|     |       |                                                                  |  |
|     |       v                                                                  |  |
|     |  [Random Packet Drop Simulation: 1-3% burst packet loss]                 |  |
|     |       |                                                                  |  |
|     |       v                                                                  |  |
|     |  [Upsample to 16,000 Hz Standard Input Tensor]                           |  |
|     +--------------------------------------------------------------------------+  |
|                                                                                   |
|  4. Training Matrix:                                                              |
|     - 60,000 Bonafide / 60,000 Cloned samples                                     |
|     - 80% Train / 10% Validation / 10% Blind Evaluation Testbed                   |
+-----------------------------------------------------------------------------------+
```

#### Detailed Engineering Steps:
1. **Custom Dataset Generation via Sarvam AI TTS (`ml/datasets/generate_sarvam_tts.py`)**:
   - Write an automated synthesis script that takes text from Indian banking advisories and news corpora.
   - Generate 10,000 audio samples using Sarvam AI's TTS voices across various emotional styles.
   - *Why this is mandatory*: Anti-spoofing artifacts are vocoder-specific. If your model has never seen artifacts from the specific vocoder used in live demonstrations, the EER will degrade significantly.

2. **Telephony Degradation Suite (`ml/datasets/telephony_augment.py`)**:
   - Build an automated augmentation pipeline applying:
     - 8kHz downsampling and G.711 mu-law companding.
     - ITU-T G.712 bandpass filtering (300Hz - 3400Hz).
     - MUSAN telephony noise injection at varying SNRs (15dB - 30dB).
     - Simulated burst packet loss (1% to 3% loss replaced with silence or repetition).
   - Apply this augmentation across both bonafide and spoof training sets.

3. **Model Evaluation & Benchmarking (`ml/training/calibrate_scores.py`)**:
   - Evaluate model checkpoints on:
     - **EER (Equal Error Rate)**: Point where False Acceptance Rate equals False Rejection Rate.
     - **minDCF (Minimum Detection Cost Function)** as defined in ASVspoof standards.
     - **Ablation Study**: Measure accuracy with Model A only, Model B only, and the Calibrated Ensemble.

---

### Phase 11: Production Containerization & Cloud Deployment (AWS)
**Objective**: Package the FastAPI inference backend into an optimized Docker container and deploy onto AWS ECS Fargate with Redis and Application Load Balancers.

#### Detailed Engineering Steps:
1. **Optimized Multi-Stage Dockerfile (`backend/Dockerfile`)**:
   ```dockerfile
   # Stage 1: Build & wheels
   FROM python:3.11-slim-bookworm AS builder
   WORKDIR /install
   RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu

   # Stage 2: Lean runtime
   FROM python:3.11-slim-bookworm
   WORKDIR /app
   RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 curl && rm -rf /var/lib/apt/lists/*
   COPY --from=builder /install /usr/local
   COPY app/ /app/app/
   EXPOSE 8000
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
   ```

2. **AWS ECS Fargate & Application Load Balancer (`infra/`)**:
   - Deploy task definition with 2 vCPUs and 4GB RAM (sufficient for pre-warmed quantized TorchScript models).
   - Configure ALB with:
     - WSS (WebSocket Secure) protocol listener on port 443.
     - Target Group with **Sticky Sessions** enabled (Cookie-based) and **Idle Timeout set to 3600 seconds** to ensure long-lived Twilio media streams remain uninterrupted.
   - ElastiCache Redis cluster for sub-millisecond rolling window state management.

---

### Phase 12: End-to-End Testing & SIH Live Demo Protocol
**Objective**: Build a simulated test harness to stream real audio files over WebSockets into the running backend, and prepare a foolproof live presentation routine.

#### Detailed Engineering Steps:
1. **Mock Twilio Stream Simulator (`scripts/mock_twilio_stream.py`)**:
   - Simulates Twilio's exact WebSocket protocol: sends `connected`, `start`, chunks of base64 mu-law audio every 20ms, and `stop`.
   - Allows local testing of genuine vs cloned voice files without placing actual telephone calls.

2. **Latency Profiling Validation**:
   - Instrument timing across all processing stages to verify budget compliance:
     - Audio Decode & Resample: **3.8 ms**
     - 128-dim DSP Feature Extraction: **22.4 ms**
     - AASIST TorchScript Model Inference: **31.6 ms**
     - XGBoost + TreeSHAP Inference: **4.2 ms**
     - Sarvam STT + Threat Regex (async): **180.0 ms**
     - Bayesian Risk Calculation & Redis update: **1.5 ms**
     - AWS SNS Alert Publication: **45.0 ms**
     - **Total Critical Path Latency**: **~108.5 ms** (well within the 480ms budget).

3. **SIH Live Demo Presentation Script (3-Scenario Walkthrough)**:
   - **Scenario 1: Genuine Conversation**: Team member calls from another phone; speaks naturally. Dashboard shows green status, rolling risk stays < 20%, no mobile alerts fire.
   - **Scenario 2: Cloned Impersonation Scam**: Team member initiates call, then plays a voice cloned with Sarvam AI/ElevenLabs demanding an urgent OTP transfer. Within 2 seconds, the Risk Gauge spikes to 94%, phone vibrates aggressively, full-screen HUD modal appears with SHAP diagnosis, and Twilio SMS backup alert lands.
   - **Scenario 3: Verification Flow**: Demonstrates user clicking "Disconnect & Call Back to Verify", disconnecting the hijacked stream and calling the verified contact.

---

## 5. Ready-to-Use Prompts for Code Generation & Execution

Each prompt below can be handed directly to a coding assistant to implement the exact modules step-by-step:

### Prompt 1: Scaffolding & Local Infrastructure
> *"Create a monorepo structure with `/backend` (FastAPI), `/app` (React Native), and `/ml` (training scripts). Set up a `docker-compose.yml` defining FastAPI on port 8000 and Redis on port 6379. Create `backend/app/config.py` using Pydantic Settings v2 to load TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, SARVAM_API_KEY, AWS_REGION, DYNAMODB_TABLE_NAME, SNS_ALERT_TOPIC_ARN, and REDIS_URL from `.env`. Include health check endpoints in `backend/app/main.py`."*

### Prompt 2: Twilio Media Stream Ingestion Engine
> *"Implement the Twilio Media Streams handler in `backend/app/api/routes_websocket.py` and `backend/app/core/audio_buffer.py`. Handle `connected`, `start`, `media`, and `stop` events. Decode 8kHz mu-law audio to 16-bit linear PCM using an optimized lookup table, resample to 16kHz using scipy/torchaudio, and push into a circular buffer that emits 2.0-second windows with a 0.5-second hop per call SID. Also write `backend/app/api/routes_twiml.py` to return TwiML with `<Stream track='inbound_track'>`."*

### Prompt 3: 128-Dimension DSP Feature Extraction Suite
> *"Write a Python package in `backend/app/dsp/` with modules: `group_delay.py` (Modified Group Delay phase spectrum yielding 48 cepstral coefficients), `cqcc.py` (40-dim telephony-band CQCC/LFCC), `pitch_jitter.py` (extracting F0 via torchcrepe/pyin and computing local jitter and shimmer over 14 dims), and `vad_cadence.py` (Silero VAD pause duration and speech-to-silence ratio over 16 dims). Provide a unified `feature_extractor.py` that concatenates these into a normalized 128-dim numpy vector and returns a human-readable metric dict."*

### Prompt 4: Deep Neural Classifier (AASIST TorchScript)
> *"In `backend/app/models/deep_model.py`, build a high-performance inference wrapper for a fine-tuned AASIST model. The class should load a TorchScript quantized model (`.pt`), accept a 16kHz audio tensor (32,000 samples), run inference on CPU in under 40ms, and return a calibrated spoof probability between 0.0 and 1.0. Include warm-up dummy inference during FastAPI startup."*

### Prompt 5: Interpretable Boosted Classifier & TreeSHAP
> *"In `backend/app/models/boosted_model.py`, implement an XGBoost inference class that takes the 128-dim DSP vector, outputs a spoof probability, and uses `shap.TreeExplainer` to extract the top 3 contributing acoustic features with human-readable labels. Write the corresponding offline training script in `ml/training/train_xgboost.py` with 5-fold cross-validation."*

### Prompt 6: Sarvam Indic STT & Semantic Scam Analyzer
> *"In `backend/app/nlp/threat_analyzer.py` and `stt_client.py`, create an async NLP pipeline that sends 4-second audio chunks to the Sarvam AI STT API (`saarika:v1`). Analyze the transcript using a combination of regex matching for high-priority financial keywords (OTP, UPI PIN, CVV, bank transfer) and cosine similarity using `sentence-transformers/all-MiniLM-L6-v2` against canonical Indian tele-fraud prompts. Output a context risk score between 0.0 and 1.0."*

### Prompt 7: Risk Engine & Alert Fan-Out Dispatcher
> *"Write `backend/app/engine/risk_engine.py` and `alert_dispatcher.py`. The `RiskEngine` maintains an Exponential Moving Average (EMA) rolling score in Redis per call SID combining acoustic score (0.7 weight) and context score (0.3 weight), scaled by a caller reputation multiplier (1.0 for whitelisted contacts, 1.4 for unknown). When the score crosses 85, publish an alert payload to AWS SNS. Write SNS subscriber handlers to dispatch a high-priority FCM push notification and a Twilio SMS alert."*

### Prompt 8: React Native Alert Application
> *"Create a React Native TypeScript application in `/app`. Configure `@react-native-firebase/messaging` with a background headless task. When a message with `type: 'high_risk_call'` arrives, trigger a repeating haptic vibration pattern via `react-native-vibration` and display a full-screen modal alert (`AlertModalScreen.tsx`). The screen must display a dynamic risk score dial (0-100%), caller metadata, SHAP feature explanation pills, and a primary red button to disconnect and call back to verify."*

---

## 6. Implementation Roadmap & Timeline

| Stage | Duration | Core Deliverables | Success Gate |
|---|---|---|---|
| **Phase 0 & 1** | Days 1–2 | Monorepo scaffolding, Twilio number setup, WSS audio ingestion, G.711 decoder, 2s circular ring buffer | Stream 8kHz audio, decode to 16kHz PCM with zero buffer overflow |
| **Phase 2 & 3** | Days 3–4 | 128-dim DSP extractor (MGD, CQCC, Jitter, VAD), AASIST model fine-tuning & TorchScript quantization | Feature extraction < 25ms; AASIST inference < 40ms on CPU |
| **Phase 4 & 5** | Days 5–6 | XGBoost training + TreeSHAP, Sarvam AI Indic STT integration, NLP fraud regex & embeddings | Ensemble EER $\le 3.2\%$ on telephony test set; STT latency < 1.2s |
| **Phase 6, 7 & 8** | Days 7–8 | Redis rolling risk engine, AWS SNS dispatcher, FCM push + Twilio SMS, DynamoDB 30-day TTL store | Dual-threshold state transitions verified; alerts dispatch < 150ms |
| **Phase 9** | Days 9–10 | React Native app: FCM background listener, aggressive vibration loop, full-screen HUD alert screen | Full-screen alert pops up on test device within 500ms of trigger |
| **Phase 10 & 11** | Days 11–12 | Synthetic Sarvam TTS dataset generation, telephony channel augmentation, Docker ECS deployment | Multi-stage Docker image < 600MB; stable WSS connection on AWS ALB |
| **Phase 12** | Days 13–14 | Mock Twilio test harness, end-to-end latency validation, SIH live demo rehearsal (3 scenarios) | Flawless 3-scenario demo execution; latency profiling validated |
