# AI-powered voice clone detection — build plan (SIH PS 26104)

## v1 scope, confirmed

- Twilio bridges/places the call. Backend subscribes only to **`inbound_track`** on the Media Stream — the caller's voice, not the enrolled user's own voice. This is the right call: it's the only audio you need to score, it halves your bandwidth/compute, and it keeps you from having to think about retention policy for the user's own speech at all.
- Backend computes a rolling risk score continuously, for the whole call, starting in v1.
- App in v1: alert-only (vibration + popup when risk crosses threshold) + Twilio SMS as backup channel.
- App in v2: live risk-score meter shown on screen *during* the call. Since v1 already produces the continuous score, v2 here is mostly a data-channel + UI problem (streaming the existing score to the app), not a backend rebuild.
- Historical voiceprint / cross-session anomaly detection: v2, not v1.
- All stored session/risk data auto-deletes after 30 days. No raw audio persisted if avoidable — only features and risk timelines.

---

## Full technology stack

| Layer | Technology | Why |
|---|---|---|
| Telephony | Twilio Programmable Voice + TwiML | Places/bridges the call |
| Audio capture | Twilio Media Streams (`inbound_track`) | Streams caller audio over WSS in real time |
| Alerts (backup channel) | Twilio Programmable Messaging (SMS) | Fires alongside push notification |
| Backend runtime | Python 3.11 + FastAPI | WebSocket ingestion + REST APIs + ML inference in one language |
| Compute | AWS ECS Fargate (or EC2 `c6i.xlarge` if you want more control) | Long-lived process, models stay warm — avoids Lambda cold starts |
| Hot-path session state | Redis (AWS ElastiCache) | Fast rolling-average risk score per live call; DynamoDB is too slow to hit every chunk |
| Persisted session/risk log | AWS DynamoDB, TTL attribute = 30 days | Auto-deletes itself, no cron needed |
| Optional audio/feature dump | AWS S3, 30-day lifecycle rule | Only if you need it for debugging — skip if you can |
| Fan-out / alerting | AWS SNS → Firebase Cloud Messaging + Twilio SMS | One event, two channels |
| Deep detection model | PyTorch, RawNet2 or AASIST (pretrained + fine-tuned) | Option A |
| Feature-based classifier | XGBoost / LightGBM (scikit-learn ecosystem) | Option B |
| DSP feature extraction | `librosa`, `torchaudio`, `numpy`/`scipy` for STFT + group delay | Phase, spectral, pause, pitch features |
| Pitch tracking | `torchcrepe` (CREPE) or `pyin` | Jitter/micro-prosody |
| Voice activity detection | Silero VAD | Pause/rhythm segmentation |
| Speech-to-text (context layer) | Sarvam AI STT (primary), Whisper (fallback) | Transcribes call for keyword/context analysis |
| Risk-phrase detection | Rules/regex to start; `sentence-transformers` embedding similarity if you have time | Flags "OTP", "transfer", "account number" etc. |
| Mobile app | React Native | Cross-platform, matches your existing JS/TS/React skillset |
| Push notifications | Firebase Cloud Messaging | Triggers native vibration + popup on high-priority push |
| Containerization | Docker | Consistent deploy to ECS |
| CI/CD (if time allows) | GitHub Actions | Build/push/deploy on merge |

Datasets for training/eval: **ASVspoof 2019 LA / 2021 DF**, the **In-the-Wild deepfake audio dataset**, plus a **self-generated set of Sarvam AI TTS clips** (critical — your eval set must include clips from the actual vocoder you'll demo against, since anti-spoofing artifacts are vocoder-specific).

---

## Build workflow, phase by phase

Each phase lists the goal, what you're building, and a ready-to-use prompt you can hand to a coding assistant (e.g. Claude Code) to scaffold it.

### Phase 0 — Accounts & scaffolding
Set up: Twilio account + phone number, AWS account (IAM user with ECS/DynamoDB/S3/SNS/ElastiCache permissions), Sarvam AI API key, Firebase project for FCM, GitHub repo.

> **Prompt:** "Set up a monorepo with `/backend` (Python FastAPI), `/app` (React Native), and `/ml` (training scripts) folders. Add a `docker-compose.yml` for local dev with FastAPI + Redis. Add `.env.example` listing TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, SARVAM_API_KEY, AWS credentials, FCM server key."

### Phase 1 — Twilio call bridging + Media Stream ingestion
Buy a Twilio number, write the TwiML to bridge an incoming/outgoing call and start a `<Stream>` with `track="inbound_track"` pointing at your backend's WSS endpoint. Build the FastAPI WebSocket endpoint that receives base64 mu-law frames, decodes them to PCM, and buffers into ~2-second overlapping windows.

> **Prompt:** "Write a FastAPI WebSocket endpoint at `/media-stream` that implements Twilio's Media Streams protocol: handle `connected`, `start`, `media`, and `stop` events. Decode the base64 mu-law 8kHz audio payload to 16-bit PCM using `audioop` or `numpy`. Buffer decoded audio into a rolling 2-second window with 0.5s overlap per call SID, and log frame timestamps. Also generate the TwiML for a Twilio Voice webhook that bridges a call and starts a stream with track='inbound_track' only."

### Phase 2 — DSP feature extraction module
Implement the four feature families as a single Python module that takes a raw PCM chunk and returns a feature dict.

> **Prompt:** "Write a Python module `dsp_features.py` with functions: `group_delay_features(audio, sr)` computing the modified group delay function from the STFT phase spectrum; `cqcc_features(audio, sr)` computing constant-Q cepstral coefficients; `pitch_jitter(audio, sr)` using torchcrepe to extract F0 and compute frame-to-frame jitter; `pause_rhythm_stats(audio, sr)` using Silero VAD to compute pause duration distribution and speech rate. Each function should return a fixed-length numpy feature vector plus a human-readable summary dict for the risk explanation panel."

### Phase 3 — Detection model A (deep model)
Get a pretrained RawNet2 or AASIST checkpoint running for inference, then fine-tune on ASVspoof + your Sarvam-generated clips.

> **Prompt:** "Set up an inference wrapper around a pretrained AASIST model (or RawNet2) that takes a raw audio chunk and returns a spoof probability [0,1]. Then write a fine-tuning script using ASVspoof 2019 LA training data plus a custom directory of additional bonafide/spoof clips, with early stopping on EER. Export the fine-tuned model as a TorchScript file for fast inference in the FastAPI service."

### Phase 4 — Detection model B (feature-based classifier)
Train XGBoost on the DSP features from Phase 2, labeled bonafide/spoof.

> **Prompt:** "Write a training script that loads DSP feature vectors (group delay, CQCC, pitch jitter, pause stats) with bonafide/spoof labels, trains an XGBoost binary classifier, reports precision/recall/EER, and saves the model. Also write the corresponding inference function for use in the live pipeline."

### Phase 5 — Ensemble + context/NLP layer
Combine Model A + Model B scores, and add the transcription/keyword layer.

> **Prompt:** "Write an ensemble function that takes Model A's spoof probability and Model B's spoof probability and combines them (start with a simple weighted average, expose weights as config). Separately, write a `context_analyzer.py` that calls Sarvam AI's STT on a buffered audio segment, then scans the transcript against a configurable list of risk phrases (grouped by severity: high = OTP/password/UPI PIN, medium = transfer/account number/urgent) and returns a context risk score."

### Phase 6 — Risk scoring engine
Combine everything into the final score with the caller-reputation multiplier and rolling average.

> **Prompt:** "Write a `RiskEngine` class, one instance per active call SID, backed by Redis. On each new chunk it should: pull acoustic_score from the ensemble, pull context_score from the NLP layer, apply a caller_multiplier (1.5x if the caller number isn't in the enrolled contacts list, 1.0x if known), maintain a rolling average over the last N chunks, and expose `get_current_risk()` and `check_threshold_crossed()`. Thresholds: medium=70, high=85."

### Phase 7 — Alerting
Wire threshold crossings to SNS, fanning out to FCM push and Twilio SMS.

> **Prompt:** "Write an `AlertDispatcher` that, when RiskEngine reports a threshold crossing, publishes an SNS event with call SID, risk score, and top contributing signals. Add two SNS subscribers: one that sends a high-priority FCM push notification (title, body with risk explanation, custom data payload for the app to trigger vibration) and one that sends a Twilio SMS with a short warning message."

### Phase 8 — Data layer
DynamoDB schema with TTL, no raw audio.

> **Prompt:** "Design a DynamoDB table `call_sessions` with primary key `call_sid`, attributes for start time, caller number, rolling risk score history (list of {timestamp, score}), final verdict, and a `ttl` attribute set to now + 30 days. Write the read/write functions from the FastAPI service."

### Phase 9 — Mobile app (v1: alert-only)
React Native app that registers for FCM and shows a vibration + full-screen popup on a high-risk push.

> **Prompt:** "Build a React Native app with Firebase Cloud Messaging integration. On receiving a push notification with `type: high_risk_call` in its data payload, trigger `Vibration.vibrate([...])` and show a full-screen modal alert with the risk score and reason, plus a 'Call back to verify' button. Include a simple call history screen listing past alerts from a REST endpoint."

### Phase 10 — Deployment
Containerize and deploy the backend.

> **Prompt:** "Write a Dockerfile for the FastAPI backend (multi-stage build, install torch CPU-only wheel to keep image size down). Write an ECS Fargate task definition and service, an Application Load Balancer with a WSS-compatible target group, and the IAM role permissions needed for DynamoDB, SNS, and ElastiCache access."

### Phase 11 — Testing & demo prep
Generate Sarvam TTS eval clips, calibrate thresholds, rehearse the live demo (genuine call → cloned call → voiceprint-style impersonation claim).

---

## v2 roadmap (not now, just so it's written down)

- Historical voiceprint matching: enroll the genuine user once via a controlled recording in the app (not by capturing their live call audio), store an embedding (ECAPA-TDNN/Resemblyzer), compare cosine similarity against live calls claiming to be that person.
- Live in-app risk meter during the call: expose the RiskEngine's rolling score over a lightweight endpoint (WebSocket or short-poll) that the app subscribes to while a call is active.
