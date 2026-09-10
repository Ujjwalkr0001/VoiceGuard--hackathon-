# VoiceGuard — Detailed Implementation Plan (SIH PS 26104)

> AI-Powered Voice Clone Detection System — Step-by-step build checklist.
> Mark each step `[x]` when completed. Steps marked `[/]` are in progress.

---

## Phase 0 — Accounts, Credentials & Project Scaffolding

### 0.1 — External Account Setup

- [ ] **Step 1:** Create a Twilio account and verify email/phone
- [ ] **Step 2:** Purchase a Twilio phone number with Voice + SMS capability
- [ ] **Step 3:** Note down `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`
- [ ] **Step 4:** Create an AWS account (or use existing) and set up IAM user with programmatic access
- [ ] **Step 5:** Attach IAM policies for ECS, DynamoDB, S3, SNS, ElastiCache, CloudWatch
- [ ] **Step 6:** Generate AWS Access Key ID and Secret Access Key for the IAM user
- [ ] **Step 7:** Sign up for Sarvam AI API and obtain `SARVAM_API_KEY`
- [ ] **Step 8:** Create a Firebase project in Firebase Console
- [ ] **Step 9:** Enable Firebase Cloud Messaging (FCM) in the Firebase project
- [ ] **Step 10:** Download `google-services.json` (Android) and `GoogleService-Info.plist` (iOS)
- [ ] **Step 11:** Generate FCM Server Key for backend push notifications
- [ ] **Step 12:** Create a GitHub repository for the VoiceGuard project

### 0.2 — Monorepo Structure

- [ ] **Step 13:** Initialize git repo with `main` and `dev` branches
- [ ] **Step 14:** Create `/backend` directory (Python FastAPI service)
- [ ] **Step 15:** Create `/app` directory (React Native mobile app)
- [ ] **Step 16:** Create `/ml` directory (training scripts, model artifacts, dataset tools)
- [ ] **Step 17:** Create `/infra` directory (Dockerfiles, ECS task defs, IaC configs)
- [ ] **Step 18:** Create `/docs` directory (architecture diagrams, API docs, demo scripts)

### 0.3 — Backend Project Initialization

- [ ] **Step 19:** Create `/backend/pyproject.toml` with Python 3.11 and all dependencies:
  - `fastapi`, `uvicorn[standard]`, `websockets`
  - `torch`, `torchaudio`, `torchcrepe`
  - `librosa`, `numpy`, `scipy`
  - `xgboost`, `scikit-learn`
  - `redis`, `boto3`
  - `twilio`
  - `silero-vad`
  - `sentence-transformers`
  - `python-dotenv`
- [ ] **Step 20:** Create `/backend/requirements.txt` as a pinned lockfile for Docker reproducibility
- [ ] **Step 21:** Create `/backend/.env.example` listing all required environment variables:
  ```
  TWILIO_ACCOUNT_SID=
  TWILIO_AUTH_TOKEN=
  TWILIO_PHONE_NUMBER=
  SARVAM_API_KEY=
  AWS_ACCESS_KEY_ID=
  AWS_SECRET_ACCESS_KEY=
  AWS_REGION=
  REDIS_URL=
  DYNAMODB_TABLE_NAME=
  SNS_TOPIC_ARN=
  FCM_SERVER_KEY=
  RISK_THRESHOLD_MEDIUM=70
  RISK_THRESHOLD_HIGH=85
  ```
- [ ] **Step 22:** Create `/backend/app/__init__.py` and `/backend/app/main.py` with a bare FastAPI app skeleton
- [ ] **Step 23:** Create `/backend/app/config.py` — load all env vars with validation using Pydantic `BaseSettings`

### 0.4 — Docker & Local Dev Environment

- [ ] **Step 24:** Create `docker-compose.yml` at project root with services:
  - `backend` (FastAPI, ports 8000, hot-reload mount)
  - `redis` (Redis 7, port 6379)
- [ ] **Step 25:** Create `/backend/Dockerfile.dev` for local development with hot-reload
- [ ] **Step 26:** Verify `docker compose up` starts both services and FastAPI responds on `http://localhost:8000/health`
- [ ] **Step 27:** Create a `/backend/app/health.py` health check endpoint that pings Redis and returns status

### 0.5 — Logging, Error Handling & Project Conventions

- [ ] **Step 28:** Set up structured JSON logging using Python `logging` + `structlog`
- [ ] **Step 29:** Create `/backend/app/exceptions.py` — define custom exception classes for audio processing errors, model inference errors, and external service failures
- [ ] **Step 30:** Add global exception handlers in FastAPI for graceful error responses
- [ ] **Step 31:** Create `.gitignore` covering Python, Node, env files, model checkpoints, datasets

---

## Phase 1 — Twilio Call Bridging + Media Stream Ingestion

### 1.1 — TwiML & Webhook Setup

- [ ] **Step 32:** Create `/backend/app/routes/twilio_webhook.py` — POST endpoint for Twilio Voice webhook
- [ ] **Step 33:** Implement TwiML response that `<Dial>`s to the target number and includes `<Stream url="wss://your-host/media-stream" track="inbound_track" />`
- [ ] **Step 34:** Configure Twilio phone number's Voice webhook to point to the FastAPI endpoint (use ngrok for local dev)
- [ ] **Step 35:** Test that dialing the Twilio number establishes a bridged call and logs a connection

### 1.2 — WebSocket Media Stream Endpoint

- [ ] **Step 36:** Create `/backend/app/routes/media_stream.py` — FastAPI WebSocket endpoint at `/media-stream`
- [ ] **Step 37:** Implement Twilio Media Streams protocol handler: parse `connected`, `start`, `media`, `stop` JSON events
- [ ] **Step 38:** Extract `streamSid`, `callSid`, `track` from the `start` event and log them
- [ ] **Step 39:** Decode base64 mu-law 8 kHz audio payload from `media` events to 16-bit PCM using `audioop.ulaw2lin`
- [ ] **Step 40:** Implement a per-call audio buffer manager class (`AudioBufferManager`) that accumulates decoded PCM frames

### 1.3 — Windowing & Chunk Dispatch

- [ ] **Step 41:** Implement rolling 2-second window with 0.5-second overlap (hop) in `AudioBufferManager`
- [ ] **Step 42:** When a full 2-second window is ready, emit the chunk to the processing pipeline (asyncio queue)
- [ ] **Step 43:** Add frame-timestamp tracking for each chunk to enable time-aligned risk scoring
- [ ] **Step 44:** Write unit tests for `AudioBufferManager`: verify correct chunk size, overlap, and timestamp alignment
- [ ] **Step 45:** End-to-end test: make a real Twilio call, verify chunks are produced and logged with correct durations

---

## Phase 2 — DSP Feature Extraction Module

### 2.1 — Group Delay Features

- [ ] **Step 46:** Create `/backend/app/ml/dsp_features.py`
- [ ] **Step 47:** Implement `group_delay_features(audio: np.ndarray, sr: int)`:
  - Compute STFT of the audio signal
  - Extract phase spectrum
  - Compute modified group delay function (MODGDF) per Hegde et al.
  - Apply cepstral smoothing to suppress noise
  - Return fixed-length feature vector (e.g., 20-dim mean/std of group delay across frequency bins)
- [ ] **Step 48:** Add human-readable summary dict (e.g., `{"group_delay_deviation": 0.42, "phase_coherence": 0.78}`)
- [ ] **Step 49:** Write unit test with a known bonafide WAV and a known spoof WAV to verify feature separation

### 2.2 — Constant-Q Cepstral Coefficients (CQCC)

- [ ] **Step 50:** Implement `cqcc_features(audio: np.ndarray, sr: int)`:
  - Compute Constant-Q Transform (CQT) using `librosa.cqt`
  - Apply log-power spectrum
  - Compute DCT to produce cepstral coefficients
  - Extract first 20 CQCCs + deltas + double-deltas (60-dim total)
- [ ] **Step 51:** Return fixed-length numpy vector plus summary dict
- [ ] **Step 52:** Write unit test comparing CQCC distributions between bonafide and TTS audio samples

### 2.3 — Pitch Jitter & Micro-Prosody

- [ ] **Step 53:** Implement `pitch_jitter(audio: np.ndarray, sr: int)`:
  - Use `torchcrepe.predict` to extract F0 contour at 10ms resolution
  - Compute frame-to-frame jitter (absolute and relative)
  - Compute shimmer (amplitude perturbation)
  - Compute F0 statistics: mean, std, range, slope
  - Return feature vector (~8-dim) plus summary
- [ ] **Step 54:** Handle edge cases: silence, very short chunks, unvoiced segments
- [ ] **Step 55:** Write unit test: verify jitter is near-zero for constant-pitch synthetic tone, higher for natural speech

### 2.4 — Pause/Rhythm Statistics (VAD-based)

- [ ] **Step 56:** Implement `pause_rhythm_stats(audio: np.ndarray, sr: int)`:
  - Load Silero VAD model (cache globally to avoid reloading)
  - Run VAD to get speech/silence segments with timestamps
  - Compute: number of pauses, mean pause duration, max pause duration, pause duration variance
  - Compute: speech rate (voiced frames / total frames), speech segment mean duration
  - Return feature vector (~8-dim) plus summary
- [ ] **Step 57:** Write unit test with audio containing known pauses
- [ ] **Step 58:** Benchmark Silero VAD inference time on 2-second chunks (must be < 50ms)

### 2.5 — Unified Feature Pipeline

- [ ] **Step 59:** Create `extract_all_features(audio: np.ndarray, sr: int) -> dict` that calls all four functions and concatenates into a single feature vector
- [ ] **Step 60:** Add feature normalization (z-score) based on precomputed mean/std from training data
- [ ] **Step 61:** Add timing instrumentation — log time taken by each feature function per chunk
- [ ] **Step 62:** Integration test: feed a full 2-second PCM chunk from Twilio through the entire feature pipeline, verify output shape and no NaN/Inf values

---

## Phase 3 — Detection Model A: Deep Neural Network (AASIST / RawNet2)

### 3.1 — Dataset Preparation

- [ ] **Step 63:** Download ASVspoof 2019 LA dataset (train + dev + eval partitions)
- [ ] **Step 64:** Download ASVspoof 2021 DF dataset (for cross-dataset evaluation)
- [ ] **Step 65:** Download the In-the-Wild deepfake audio dataset
- [ ] **Step 66:** Generate 200+ synthetic speech clips using Sarvam AI TTS API (multiple voices, multiple texts, Hindi + English)
- [ ] **Step 67:** Generate additional synthetic clips with other publicly available TTS systems (Coqui, Bark, etc.) for diversity
- [ ] **Step 68:** Create `/ml/data/` directory structure: `train/bonafide/`, `train/spoof/`, `eval/bonafide/`, `eval/spoof/`
- [ ] **Step 69:** Write a data preprocessing script (`/ml/scripts/prepare_dataset.py`) that:
  - Resamples all audio to 16 kHz mono
  - Trims silence from start/end
  - Normalizes amplitude
  - Splits into train/eval with stratification
- [ ] **Step 70:** Create a manifest CSV (`path, label, source, duration`) for all datasets combined

### 3.2 — Pretrained Model Setup

- [ ] **Step 71:** Clone the official AASIST repository (or RawNet2 if chosen)
- [ ] **Step 72:** Download the pretrained checkpoint (ASVspoof 2019 LA)
- [ ] **Step 73:** Create `/ml/models/aasist_wrapper.py` — inference wrapper class:
  - `load_model(checkpoint_path)` — loads weights, sets eval mode
  - `predict(audio: np.ndarray) -> float` — returns spoof probability [0, 1]
- [ ] **Step 74:** Test inference on 10 bonafide + 10 spoof samples, verify reasonable separation in scores
- [ ] **Step 75:** Benchmark single-chunk inference latency (target: < 100ms on CPU for 2-second audio)

### 3.3 — Fine-Tuning

- [ ] **Step 76:** Write `/ml/scripts/finetune_aasist.py`:
  - Custom PyTorch `Dataset` class that loads audio and returns (waveform, label) pairs
  - Data augmentation: additive noise (SNR 10-30dB), room impulse response convolution, telephone band-pass filter (300-3400 Hz to simulate Twilio)
  - Training loop with Adam optimizer, cosine annealing LR scheduler
  - Validation after each epoch: compute EER, min t-DCF
  - Early stopping on EER (patience = 5 epochs)
- [ ] **Step 77:** Run fine-tuning for up to 30 epochs on the combined dataset
- [ ] **Step 78:** Evaluate fine-tuned model on held-out eval set — record EER, precision, recall, F1
- [ ] **Step 79:** Evaluate specifically on Sarvam AI TTS clips — ensure the model detects them reliably
- [ ] **Step 80:** Export the best checkpoint as TorchScript (`.pt`) for deployment: `torch.jit.script(model)`
- [ ] **Step 81:** Copy the exported model to `/backend/app/ml/models/aasist_finetuned.pt`

---

## Phase 4 — Detection Model B: Feature-Based Classifier (XGBoost)

### 4.1 — Feature Extraction at Scale

- [ ] **Step 82:** Write `/ml/scripts/extract_features_bulk.py`:
  - Iterate over all audio files in the dataset manifest
  - Call `extract_all_features()` from Phase 2 on each file
  - Save feature vectors + labels as a `.parquet` or `.npz` file
- [ ] **Step 83:** Run bulk extraction on the full training set (~50k+ samples)
- [ ] **Step 84:** Inspect feature distributions: plot histograms, check for NaN/Inf, verify class balance

### 4.2 — XGBoost Training

- [ ] **Step 85:** Write `/ml/scripts/train_xgboost.py`:
  - Load feature matrix and labels
  - Train/val split (80/20, stratified)
  - Train XGBoost binary classifier with hyperparameter grid search (max_depth, n_estimators, learning_rate, min_child_weight)
  - Use 5-fold cross-validation for hyperparameter selection
- [ ] **Step 86:** Report metrics: EER, precision, recall, F1, AUC-ROC, confusion matrix
- [ ] **Step 87:** Analyze feature importances — identify which DSP features contribute most
- [ ] **Step 88:** Save the trained model as a joblib/pickle file
- [ ] **Step 89:** Write `/backend/app/ml/xgboost_inference.py` — inference function:
  - `predict_spoof_probability(features: np.ndarray) -> float`
  - Load model once at startup, cache in memory

### 4.3 — Cross-Validation & Robustness

- [ ] **Step 90:** Evaluate XGBoost specifically on telephone-band filtered audio (simulating Twilio's 8 kHz mu-law)
- [ ] **Step 91:** Evaluate on Sarvam AI TTS clips — compare performance with Model A
- [ ] **Step 92:** Test on In-the-Wild dataset to check generalization beyond ASVspoof

---

## Phase 5 — Ensemble + Context/NLP Layer

### 5.1 — Ensemble Combiner

- [ ] **Step 93:** Create `/backend/app/ml/ensemble.py`
- [ ] **Step 94:** Implement `EnsembleScorer` class:
  - Takes `model_a_score` (AASIST spoof probability) and `model_b_score` (XGBoost spoof probability)
  - Combines via configurable weighted average (default: 0.6 × Model A + 0.4 × Model B)
  - Weights exposed as environment variables for easy tuning
  - Returns combined `acoustic_score` in [0, 100] range
- [ ] **Step 95:** Implement confidence-weighted fusion: if one model's output has low confidence (near 0.5), reduce its weight
- [ ] **Step 96:** Write unit tests for ensemble: verify score range, verify that strong agreement between models produces high/low scores

### 5.2 — Speech-to-Text Integration

- [ ] **Step 97:** Create `/backend/app/services/stt_service.py`
- [ ] **Step 98:** Implement Sarvam AI STT API client:
  - Accept a PCM audio buffer (2-second chunk or accumulated longer segment)
  - POST to Sarvam AI STT endpoint with correct audio format headers
  - Return transcript text + language detected + confidence
- [ ] **Step 99:** Implement Whisper fallback: if Sarvam API fails or times out (>2s), fall back to local Whisper `tiny` or `base` model
- [ ] **Step 100:** Handle Hindi, English, and code-mixed (Hinglish) transcription
- [ ] **Step 101:** Accumulate transcripts across chunks for the same call (sliding text window of last 30 seconds)

### 5.3 — Context/Risk-Phrase Analyzer

- [ ] **Step 102:** Create `/backend/app/ml/context_analyzer.py`
- [ ] **Step 103:** Define configurable risk phrase dictionary with severity tiers:
  - **Critical (weight 1.0):** OTP, password, UPI PIN, CVV, one-time password, verification code, mPIN, Aadhaar number
  - **High (weight 0.7):** transfer money, bank account, NEFT, RTGS, IMPS, send money, credit card number
  - **Medium (weight 0.4):** urgent, immediately, deadline, penalty, block account, suspend, verify identity, KYC
  - Include Hindi equivalents for all phrases
- [ ] **Step 104:** Implement regex + keyword matching for fast phrase detection
- [ ] **Step 105:** (Optional) Implement `sentence-transformers` embedding similarity for fuzzy matching of paraphrased risk phrases
- [ ] **Step 106:** Compute `context_risk_score` based on density and severity of detected phrases in the recent transcript window
- [ ] **Step 107:** Return detected phrases with timestamps for the risk explanation panel
- [ ] **Step 108:** Write unit tests with sample transcripts containing risk phrases in English, Hindi, and Hinglish

---

## Phase 6 — Risk Scoring Engine

### 6.1 — Core Risk Engine

- [ ] **Step 109:** Create `/backend/app/services/risk_engine.py`
- [ ] **Step 110:** Implement `RiskEngine` class (one instance per active call SID):
  - Attributes: `call_sid`, `caller_number`, `start_time`, `chunk_scores: list`, `rolling_window_size: int = 10`
  - On each new chunk, receive: `acoustic_score` (from ensemble), `context_score` (from NLP)
- [ ] **Step 111:** Implement `caller_multiplier` logic:
  - If caller number is in the user's enrolled contacts list → multiplier = 1.0
  - If caller number is unknown → multiplier = 1.3
  - If caller number is flagged (previously high-risk) → multiplier = 1.5
- [ ] **Step 112:** Implement composite score calculation:
  ```
  raw_score = (0.65 × acoustic_score) + (0.35 × context_score)
  adjusted_score = min(100, raw_score × caller_multiplier)
  ```
- [ ] **Step 113:** Implement rolling average over last N chunks (configurable, default 10) for temporal smoothing
- [ ] **Step 114:** Implement spike detection: if a single chunk score exceeds 95, bypass rolling average and immediately flag

### 6.2 — Redis-Backed State

- [ ] **Step 115:** Create `/backend/app/services/redis_client.py` — async Redis connection pool using `aioredis`
- [ ] **Step 116:** Store per-call state in Redis:
  - Key: `call:{call_sid}:risk` → hash with `current_score`, `peak_score`, `chunk_count`, `last_updated`
  - Key: `call:{call_sid}:history` → sorted set of `(timestamp, score)` pairs
  - TTL: 24 hours (cleaned up after call ends or on expiry)
- [ ] **Step 117:** Implement `get_current_risk(call_sid) -> float` — reads from Redis
- [ ] **Step 118:** Implement `check_threshold_crossed(call_sid) -> Optional[str]` — returns `"medium"`, `"high"`, or `None`
- [ ] **Step 119:** Implement threshold hysteresis: once a high alert fires, don't fire again unless score drops below 75 and re-crosses 85 (prevents alert spam)

### 6.3 — Risk Engine Tests

- [ ] **Step 120:** Unit test: feed a sequence of gradually increasing scores, verify `medium` threshold crossing detected at correct chunk
- [ ] **Step 121:** Unit test: verify rolling average smooths out a single spike below threshold
- [ ] **Step 122:** Unit test: verify spike detection bypasses rolling average for extreme scores
- [ ] **Step 123:** Unit test: verify hysteresis prevents duplicate alerts

---

## Phase 7 — Alerting System

### 7.1 — SNS Topic & Event Publishing

- [ ] **Step 124:** Create AWS SNS topic `voiceguard-alerts` via AWS CLI or console
- [ ] **Step 125:** Create `/backend/app/services/alert_dispatcher.py`
- [ ] **Step 126:** Implement `AlertDispatcher` class:
  - Method `dispatch_alert(call_sid, risk_score, risk_level, contributing_signals: dict)`:
    - Construct alert payload JSON: `{call_sid, risk_score, risk_level, signals, timestamp, caller_number}`
    - Publish to SNS topic

### 7.2 — FCM Push Notification Subscriber

- [ ] **Step 127:** Create an AWS Lambda (or handle inline in backend) that subscribes to the SNS topic
- [ ] **Step 128:** On SNS event, send FCM push notification:
  - Priority: `high`
  - Title: `"⚠️ VoiceGuard Alert"`
  - Body: `"Risk Level: {HIGH/MEDIUM} — {top_signal_reason}"`
  - Data payload: `{type: "high_risk_call", call_sid, risk_score, signals}`
  - Android channel: high-importance (for heads-up notification + vibration)
- [ ] **Step 129:** Test FCM delivery to a real Android device using a test payload

### 7.3 — Twilio SMS Backup Alert

- [ ] **Step 130:** Add SMS dispatch to `AlertDispatcher`:
  - On `high` risk level only (not medium, to avoid SMS spam)
  - Send Twilio SMS to enrolled user's phone number
  - Message: `"[VoiceGuard] HIGH RISK detected on your current call (score: {score}). The caller's voice shows signs of AI generation. Verify the caller's identity before sharing sensitive information."`
- [ ] **Step 131:** Test SMS delivery end-to-end

### 7.4 — Alert Rate Limiting & Deduplication

- [ ] **Step 132:** Implement per-call alert cooldown: max 1 alert per 60 seconds per call SID
- [ ] **Step 133:** Store last alert timestamp in Redis to enforce cooldown
- [ ] **Step 134:** Log all dispatched alerts for audit trail

---

## Phase 8 — Data Persistence Layer (DynamoDB)

### 8.1 — DynamoDB Table Design

- [ ] **Step 135:** Create DynamoDB table `call_sessions`:
  - Partition key: `call_sid` (String)
  - Attributes:
    - `caller_number` (String)
    - `user_id` (String)
    - `start_time` (Number — Unix epoch)
    - `end_time` (Number — Unix epoch)
    - `duration_seconds` (Number)
    - `risk_score_timeline` (List of Maps: `[{timestamp, score, signals}]`)
    - `peak_risk_score` (Number)
    - `final_verdict` (String: `safe`, `suspicious`, `high_risk`)
    - `alerts_sent` (List of Maps: `[{timestamp, level, channel}]`)
    - `ttl` (Number — Unix epoch, set to `start_time + 30 days`)
  - Enable TTL on the `ttl` attribute for automatic 30-day deletion

### 8.2 — Data Access Functions

- [ ] **Step 136:** Create `/backend/app/services/dynamo_client.py`
- [ ] **Step 137:** Implement `create_session(call_sid, caller_number, user_id)` — inserts a new record at call start
- [ ] **Step 138:** Implement `append_risk_score(call_sid, timestamp, score, signals)` — appends to `risk_score_timeline`
- [ ] **Step 139:** Implement `finalize_session(call_sid, final_verdict)` — sets end_time, duration, peak score
- [ ] **Step 140:** Implement `get_session(call_sid) -> dict` — retrieves full session record
- [ ] **Step 141:** Implement `list_sessions(user_id, limit=20) -> list` — returns recent sessions for a user, sorted by start_time descending
- [ ] **Step 142:** Write integration tests against DynamoDB Local (Docker container)

---

## Phase 9 — Live Processing Pipeline Orchestration

### 9.1 — Pipeline Coordinator

- [ ] **Step 143:** Create `/backend/app/pipeline/coordinator.py`
- [ ] **Step 144:** Implement `CallPipelineCoordinator` class that ties all components together:
  - Instantiated when a new Twilio Media Stream connects
  - Holds references to: `AudioBufferManager`, `RiskEngine`, `AlertDispatcher`, `DynamoClient`
  - Runs an async processing loop consuming chunks from the audio buffer queue
- [ ] **Step 145:** For each chunk, orchestrate the processing pipeline:
  1. Extract DSP features (`extract_all_features`)
  2. Run Model A inference (AASIST)
  3. Run Model B inference (XGBoost on DSP features)
  4. Compute ensemble score
  5. Run STT (async, batched — accumulate 5 seconds of audio before transcribing)
  6. Run context analysis on accumulated transcript
  7. Feed scores to RiskEngine
  8. Check threshold → dispatch alert if crossed
  9. Persist risk score to DynamoDB (batch, every 5th chunk to reduce writes)
- [ ] **Step 146:** Implement graceful cleanup on call end (`stop` event): finalize DynamoDB session, flush Redis state
- [ ] **Step 147:** Implement concurrent model inference: run Model A and Model B in parallel using `asyncio.gather` or thread pool

### 9.2 — Performance Optimization

- [ ] **Step 148:** Profile end-to-end pipeline latency per chunk (target: < 500ms total)
- [ ] **Step 149:** Optimize model loading: load AASIST and XGBoost models once at FastAPI startup, share across all call sessions
- [ ] **Step 150:** Implement model warm-up: run a dummy inference at startup to JIT-compile and warm caches
- [ ] **Step 151:** Add circuit breaker for Sarvam AI STT: if 3 consecutive failures, switch to Whisper fallback for 60 seconds
- [ ] **Step 152:** Add metrics collection: per-chunk latency, model inference time, feature extraction time (log to CloudWatch or stdout for now)

---

## Phase 10 — REST API Endpoints

### 10.1 — Call History API

- [ ] **Step 153:** Create `/backend/app/routes/api.py`
- [ ] **Step 154:** Implement `GET /api/v1/calls` — list recent calls for a user (paginated)
- [ ] **Step 155:** Implement `GET /api/v1/calls/{call_sid}` — get full call session details including risk timeline
- [ ] **Step 156:** Implement `GET /api/v1/calls/{call_sid}/risk-timeline` — return risk scores over time (for v2 live chart)

### 10.2 — Contacts / Trusted Numbers API

- [ ] **Step 157:** Implement `POST /api/v1/contacts` — add a trusted phone number (reduces caller_multiplier)
- [ ] **Step 158:** Implement `GET /api/v1/contacts` — list trusted contacts
- [ ] **Step 159:** Implement `DELETE /api/v1/contacts/{number}` — remove a trusted contact

### 10.3 — Configuration API

- [ ] **Step 160:** Implement `GET /api/v1/config` — return current thresholds, ensemble weights
- [ ] **Step 161:** Implement `PATCH /api/v1/config` — update thresholds at runtime (for demo tuning)

### 10.4 — API Security & Validation

- [ ] **Step 162:** Add API key authentication middleware for all `/api/v1/*` routes
- [ ] **Step 163:** Add request validation using Pydantic models for all request/response schemas
- [ ] **Step 164:** Add rate limiting (10 req/s per API key)
- [ ] **Step 165:** Write API tests using `httpx` + FastAPI's `TestClient`

---

## Phase 11 — Mobile App (React Native — v1 Alert-Only)

### 11.1 — React Native Project Setup

- [ ] **Step 166:** Initialize React Native project in `/app` using `npx react-native init VoiceGuardApp`
- [ ] **Step 167:** Install dependencies: `@react-native-firebase/app`, `@react-native-firebase/messaging`, `react-navigation`, `axios`
- [ ] **Step 168:** Configure Firebase: add `google-services.json` to Android, `GoogleService-Info.plist` to iOS
- [ ] **Step 169:** Set up navigation structure: Stack navigator with `Home`, `CallHistory`, `CallDetail`, `Settings` screens

### 11.2 — FCM Push Notification Handling

- [ ] **Step 170:** Register for FCM push tokens on app startup, send token to backend
- [ ] **Step 171:** Implement foreground notification handler: show in-app alert when a push arrives while app is open
- [ ] **Step 172:** Implement background notification handler: trigger system notification with high priority
- [ ] **Step 173:** On receiving `type: "high_risk_call"` data payload:
  - Trigger `Vibration.vibrate([500, 200, 500, 200, 500])` pattern
  - Navigate to full-screen alert modal

### 11.3 — Alert Modal UI

- [ ] **Step 174:** Create `HighRiskAlertModal` component:
  - Full-screen red/amber gradient overlay
  - Large warning icon with pulse animation
  - Risk score displayed prominently
  - Contributing signals listed (e.g., "AI-generated voice patterns detected", "Caller asked for OTP")
  - "Dismiss" button and "End Call" button
- [ ] **Step 175:** Add haptic feedback on modal appearance

### 11.4 — Call History Screen

- [ ] **Step 176:** Create `CallHistoryScreen`:
  - Fetch `GET /api/v1/calls` from backend
  - Display list of recent calls with: caller number, timestamp, final verdict (color-coded badge: green/yellow/red)
  - Pull-to-refresh
- [ ] **Step 177:** Create `CallDetailScreen`:
  - Fetch `GET /api/v1/calls/{call_sid}` from backend
  - Show risk score timeline as a simple line chart
  - Show alerts that were fired during the call
  - Show detected risk phrases from transcript

### 11.5 — Settings Screen

- [ ] **Step 178:** Create `SettingsScreen`:
  - Toggle notifications on/off
  - Manage trusted contacts (add/remove phone numbers)
  - Display app version, privacy policy link

### 11.6 — App Polish

- [ ] **Step 179:** Apply consistent design system: dark theme, VoiceGuard branding colors (deep blue + amber accents)
- [ ] **Step 180:** Add app icon and splash screen
- [ ] **Step 181:** Test on Android emulator + physical device
- [ ] **Step 182:** Test push notification delivery in background, killed, and foreground states

---

## Phase 12 — Deployment (Containerization + AWS)

### 12.1 — Production Dockerfile

- [ ] **Step 183:** Create `/backend/Dockerfile` (multi-stage build):
  - Stage 1: Install Python dependencies including PyTorch CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`)
  - Stage 2: Copy app code + model artifacts, set entrypoint to `uvicorn app.main:app`
  - Target image size: < 2 GB
- [ ] **Step 184:** Build and test Docker image locally: verify health check, WebSocket connection, model inference all work

### 12.2 — AWS Infrastructure

- [ ] **Step 185:** Create ECR repository `voiceguard-backend`, push Docker image
- [ ] **Step 186:** Create ECS Fargate cluster `voiceguard-cluster`
- [ ] **Step 187:** Create ECS task definition:
  - CPU: 2 vCPU, Memory: 4 GB (enough for PyTorch + XGBoost in-memory)
  - Container port: 8000
  - Environment variables from AWS Secrets Manager
  - CloudWatch log group for container logs
- [ ] **Step 188:** Create ECS service with desired count = 1 (scale later if needed)
- [ ] **Step 189:** Create Application Load Balancer (ALB) with:
  - HTTPS listener (ACM certificate)
  - WebSocket-compatible target group (stickiness enabled, idle timeout 3600s for long calls)
  - Health check path: `/health`
- [ ] **Step 190:** Create ElastiCache Redis cluster (single-node `cache.t3.micro` for v1)
- [ ] **Step 191:** Configure security groups: ALB → ECS (port 8000), ECS → Redis (port 6379), ECS → DynamoDB (VPC endpoint)
- [ ] **Step 192:** Update Twilio webhook URL and Media Stream URL to point to ALB domain

### 12.3 — CI/CD (If Time Allows)

- [ ] **Step 193:** Create `.github/workflows/deploy.yml`:
  - On push to `main`: build Docker image, push to ECR, update ECS service
  - Run linting + unit tests before build
- [ ] **Step 194:** Create `.github/workflows/test.yml`:
  - On PR: run pytest, type checking, lint

---

## Phase 13 — Testing & Quality Assurance

### 13.1 — Unit Tests

- [ ] **Step 195:** Write unit tests for all DSP feature functions (edge cases: silence, noise, very short audio)
- [ ] **Step 196:** Write unit tests for AudioBufferManager (chunk boundaries, overlap correctness)
- [ ] **Step 197:** Write unit tests for RiskEngine (score calculation, threshold detection, hysteresis)
- [ ] **Step 198:** Write unit tests for AlertDispatcher (cooldown enforcement, payload correctness)
- [ ] **Step 199:** Write unit tests for context_analyzer (phrase detection in English, Hindi, Hinglish)
- [ ] **Step 200:** Achieve ≥ 80% code coverage on backend

### 13.2 — Integration Tests

- [ ] **Step 201:** Integration test: simulate a full Twilio Media Stream WebSocket session with pre-recorded audio, verify risk scores are produced and stored in DynamoDB
- [ ] **Step 202:** Integration test: trigger a high-risk scenario, verify SNS event is published
- [ ] **Step 203:** Integration test: verify Redis state is correctly created and cleaned up after call ends

### 13.3 — Model Evaluation & Threshold Calibration

- [ ] **Step 204:** Run the full pipeline on 100 bonafide + 100 spoof audio samples (telephone-band filtered)
- [ ] **Step 205:** Plot ROC curve and DET curve for the ensemble model
- [ ] **Step 206:** Determine optimal threshold for medium (target: FPR < 10%, TPR > 80%) and high (FPR < 2%, TPR > 60%) risk levels
- [ ] **Step 207:** Update `RISK_THRESHOLD_MEDIUM` and `RISK_THRESHOLD_HIGH` in config based on calibration results
- [ ] **Step 208:** Test against 20 Sarvam AI TTS clips specifically — ensure ≥ 90% detection rate
- [ ] **Step 209:** Test against 20 real human voice samples — ensure ≥ 90% true negative rate (no false alarms)

---

## Phase 14 — Demo Preparation & Rehearsal

### 14.1 — Demo Assets

- [ ] **Step 210:** Prepare 3 demo phone numbers: one for "genuine caller", one for "AI clone attacker", one for "social engineering attacker"
- [ ] **Step 211:** Generate 5 high-quality Sarvam AI TTS clips mimicking a scam call:
  - Clip 1: "Your bank account has been compromised, please share your OTP"
  - Clip 2: "This is the RBI calling, your KYC needs immediate verification"
  - Clip 3: "Please transfer ₹10,000 to this account to avoid penalty"
  - Clip 4: A casual, non-threatening AI-generated conversation
  - Clip 5: AI voice reading out random, harmless content (tests acoustic-only detection)
- [ ] **Step 212:** Prepare genuine voice recordings of the same scripts for comparison
- [ ] **Step 213:** Create a demo script document with exact steps, timing, and expected outcomes

### 14.2 — Demo Rehearsal

- [ ] **Step 214:** Rehearsal 1 — Genuine call scenario:
  - Place a real call through the system
  - Verify risk score stays below medium threshold (green/safe)
  - Verify no alerts are triggered
- [ ] **Step 215:** Rehearsal 2 — AI clone attack scenario:
  - Play a Sarvam AI TTS clip through the call
  - Verify risk score rises above high threshold within 10-15 seconds
  - Verify FCM push notification is received on demo phone
  - Verify SMS backup alert is received
- [ ] **Step 216:** Rehearsal 3 — Social engineering scenario:
  - Use a real human voice but speak scam phrases ("share your OTP", "transfer money")
  - Verify context score rises but acoustic score stays low
  - Verify medium-level alert is triggered (NLP contribution)
- [ ] **Step 217:** Rehearsal 4 — Edge case: very noisy environment
  - Place a call with background noise
  - Verify system doesn't produce excessive false positives
- [ ] **Step 218:** Time the full demo flow — ensure it fits within the allotted presentation slot
- [ ] **Step 219:** Prepare fallback: pre-recorded screen recording of the demo in case of live technical issues

### 14.3 — Documentation & Presentation

- [ ] **Step 220:** Create architecture diagram (system-level) showing: Twilio → Backend (WSS) → ML Pipeline → Redis/DynamoDB → SNS → FCM/SMS → App
- [ ] **Step 221:** Create data flow diagram showing the journey of a single audio chunk through the pipeline
- [ ] **Step 222:** Prepare a 1-page technical summary document for judges
- [ ] **Step 223:** Document all API endpoints in a Postman collection or OpenAPI spec

---

## Phase 15 — Privacy, Compliance & Hardening

- [ ] **Step 224:** Verify no raw audio is persisted anywhere (only features and risk scores)
- [ ] **Step 225:** Verify DynamoDB TTL is correctly deleting records after 30 days
- [ ] **Step 226:** Ensure all API endpoints require authentication
- [ ] **Step 227:** Ensure all data in transit uses TLS (HTTPS/WSS)
- [ ] **Step 228:** Add privacy policy page to the app explaining what data is collected and retained
- [ ] **Step 229:** Implement data export endpoint (`GET /api/v1/user/data`) for user data portability
- [ ] **Step 230:** Implement data deletion endpoint (`DELETE /api/v1/user/data`) for right to erasure
- [ ] **Step 231:** Review all logging — ensure no PII (phone numbers, transcripts) is logged in production

---

> **Total Steps: 231**
>
> Progress: `[ 0 / 231 ]` completed
>
> Last updated: 2026-09-11
