import React, { useState, useEffect, useRef, useCallback } from "react";
import { Send, Volume2, VolumeX, MessageSquare, Activity, Settings, Zap, Sparkles, Brain, User, BotMessageSquare } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { Waveform } from "./Waveform";
import { VoiceOrb } from "./VoiceOrb";
import { MetricsPanel } from "./MetricsPanel";
import { getVoiceWebSocketUrl } from "../api";
import { useMetrics } from "../hooks";
import type { Message } from "../types";

interface ChatViewProps { sessionId: string | null; messages: Message[]; onAddMessage: (msg: Message) => void; }

/* ── AI Bot Avatar with animated neural ring ── */
const BotAvatar = () => (
  <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0 relative overflow-visible group">
    <div className="absolute inset-0 rounded-xl" style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.12), rgba(139,92,246,0.12))', border: '1px solid rgba(6,182,212,0.12)' }} />
    {/* Animated ring */}
    <svg className="absolute -inset-1 w-[calc(100%+8px)] h-[calc(100%+8px)]" viewBox="0 0 44 44">
      <circle cx="22" cy="22" r="20" fill="none" stroke="url(#avatarRingGrad)" strokeWidth="0.5" strokeDasharray="4 6" opacity="0.4">
        <animateTransform attributeName="transform" type="rotate" values="0 22 22;360 22 22" dur="8s" repeatCount="indefinite" />
      </circle>
      <defs>
        <linearGradient id="avatarRingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#a78bfa" />
        </linearGradient>
      </defs>
    </svg>
    {/* Neural mini brain */}
    <svg className="relative z-10 w-5 h-5" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="6" r="1.5" fill="#22d3ee" opacity="0.8" />
      <circle cx="7" cy="12" r="1.2" fill="#a78bfa" opacity="0.7" />
      <circle cx="17" cy="12" r="1.2" fill="#22d3ee" opacity="0.7" />
      <circle cx="12" cy="18" r="1.5" fill="#a78bfa" opacity="0.8" />
      <circle cx="12" cy="12" r="2" fill="white" opacity="0.9" />
      <line x1="12" y1="6" x2="7" y2="12" stroke="#22d3ee" strokeWidth="0.5" opacity="0.4" />
      <line x1="12" y1="6" x2="17" y2="12" stroke="#a78bfa" strokeWidth="0.5" opacity="0.4" />
      <line x1="7" y1="12" x2="12" y2="18" stroke="#a78bfa" strokeWidth="0.5" opacity="0.3" />
      <line x1="17" y1="12" x2="12" y2="18" stroke="#22d3ee" strokeWidth="0.5" opacity="0.3" />
      <line x1="7" y1="12" x2="12" y2="12" stroke="#22d3ee" strokeWidth="0.3" opacity="0.3" />
      <line x1="17" y1="12" x2="12" y2="12" stroke="#a78bfa" strokeWidth="0.3" opacity="0.3" />
    </svg>
    <div className="absolute inset-0 rounded-xl opacity-15 animate-pulse" style={{ background: 'linear-gradient(135deg, #22d3ee, #a78bfa)' }} />
  </div>
);

/* ── Typing Indicator ── */
const TypingIndicator = () => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
    className="flex items-end gap-2.5">
    <BotAvatar />
    <div className="ai-bubble-bot rounded-2xl rounded-bl-sm px-5 py-3.5">
      <div className="flex items-center gap-1.5">
        <div className="w-2 h-2 rounded-full bg-neon-400/60 typing-dot" />
        <div className="w-2 h-2 rounded-full bg-violet-400/60 typing-dot" style={{ animationDelay: '0.2s' }} />
        <div className="w-2 h-2 rounded-full bg-neon-400/60 typing-dot" style={{ animationDelay: '0.4s' }} />
      </div>
    </div>
  </motion.div>
);

export const ChatView: React.FC<ChatViewProps> = ({ sessionId, messages, onAddMessage }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [inputText, setInputText] = useState("");
  const [isMuted, setIsMuted] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [micRms, setMicRms] = useState(0);
  const [playbackRms, setPlaybackRms] = useState(0);
  const [isThinking, setIsThinking] = useState(false);
  const [streamingBotText, setStreamingBotText] = useState("");
  const [streamingUserText, setStreamingUserText] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);

  const { gpuMetrics, latencyData } = useMetrics();

  // Combined RMS: picks whichever is louder — mic input or TTS playback
  const rms = Math.max(micRms, playbackRms);

  const playbackContextRef = useRef<AudioContext | null>(null);
  const playbackAnalyserRef = useRef<AnalyserNode | null>(null);
  const playbackGainRef = useRef<GainNode | null>(null);
  const captureContextRef = useRef<AudioContext | null>(null);
  const analyserContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const streamingTextRef = useRef("");
  const streamingUserTextRef = useRef("");
  const isMutedRef = useRef(false);
  const playbackTimerRef = useRef<number | null>(null);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Keep muted ref in sync
  useEffect(() => { isMutedRef.current = isMuted; }, [isMuted]);

  // Cleanup on session change
  useEffect(() => { stopVoice(); }, [sessionId]);

  // Scroll to bottom
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, streamingBotText, streamingUserText]);

  // Refs for callbacks used inside WS handlers (avoid stale closures)
  const onAddMessageRef = useRef(onAddMessage);
  onAddMessageRef.current = onAddMessage;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  const sendControl = useCallback((obj: object) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    const json = JSON.stringify(obj);
    const bytes = new TextEncoder().encode(json);
    const frame = new Uint8Array(1 + bytes.length);
    frame[0] = 0x02;
    frame.set(bytes, 1);
    wsRef.current.send(frame);
  }, []);

  const stopAllPlayback = useCallback(() => {
    activeSourcesRef.current.forEach(src => { try { src.stop(); } catch {} });
    activeSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
  }, []);

  const playPCM = useCallback((arrayBuffer: ArrayBuffer) => {
    const ctx = playbackContextRef.current;
    if (!ctx || isMutedRef.current) return;
    const int16 = new Int16Array(arrayBuffer);
    if (int16.length === 0) return;
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
    const buffer = ctx.createBuffer(1, float32.length, 24000);
    buffer.getChannelData(0).set(float32);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    // Route through analyser for playback RMS
    if (playbackAnalyserRef.current) {
      source.connect(playbackAnalyserRef.current);
    } else {
      source.connect(ctx.destination);
    }
    const st = Math.max(ctx.currentTime, nextStartTimeRef.current);
    source.start(st);
    nextStartTimeRef.current = st + buffer.duration;
    // Track for barge-in cancellation
    activeSourcesRef.current.add(source);
    source.onended = () => activeSourcesRef.current.delete(source);
    setIsSpeaking(true);
  }, []);

  const handleGatewayMessage = useCallback((msg: any) => {
    switch (msg.type) {
      case "ready":
        break;
      case "session":
        break;
      case "segment":
        if (msg.text) {
          setStreamingUserText("");
          streamingUserTextRef.current = "";
          onAddMessageRef.current({ role: "user", content: msg.text, bargeIn: msg.barge_in || false });
        }
        break;
      case "word":
        if (msg.word) {
          streamingUserTextRef.current = streamingUserTextRef.current
            ? streamingUserTextRef.current + " " + msg.word
            : msg.word;
          setStreamingUserText(streamingUserTextRef.current);
        }
        break;
      case "thinking":
        setIsThinking(true);
        setStreamingBotText("");
        streamingTextRef.current = "";
        break;
      case "ai_token":
        setIsThinking(false);
        if (msg.token) {
          streamingTextRef.current += msg.token;
          setStreamingBotText(streamingTextRef.current);
        }
        break;
      case "ai_sentence":
        break;
      case "done": {
        const finalText = streamingTextRef.current.trim();
        if (finalText) {
          onAddMessageRef.current({ role: "agent", content: finalText });
        }
        setStreamingBotText("");
        streamingTextRef.current = "";
        setIsThinking(false);
        setIsSpeaking(false);
        break;
      }
      case "barge_in": {
        // Save interrupted bot text as a partial message
        const partialBot = streamingTextRef.current.trim();
        if (partialBot) {
          onAddMessageRef.current({ role: "agent", content: partialBot, interrupted: true });
        }
        setStreamingBotText("");
        streamingTextRef.current = "";
        // Keep streaming user text — user is still speaking
        setIsThinking(false);
        setIsSpeaking(false);
        // Stop all scheduled audio immediately
        stopAllPlayback();
        break;
      }
      case "ping":
        sendControl({ type: "pong" });
        break;
      case "history":
        if (msg.role && msg.content) {
          onAddMessageRef.current({ role: msg.role, content: msg.content });
        }
        break;
      case "error":
        console.error("Gateway error:", msg);
        break;
    }
  }, [sendControl, stopAllPlayback]);

  const handleGatewayRef = useRef(handleGatewayMessage);
  handleGatewayRef.current = handleGatewayMessage;

  const startVoice = async () => {
    if (isConnecting || !sessionId) return;
    setIsConnecting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      streamRef.current = stream;

      playbackContextRef.current = new AudioContext({ sampleRate: 24000 });
      // Set up playback analyser for TTS RMS
      const pbAnalyser = playbackContextRef.current.createAnalyser();
      pbAnalyser.fftSize = 256;
      pbAnalyser.smoothingTimeConstant = 0.5;
      pbAnalyser.connect(playbackContextRef.current.destination);
      playbackAnalyserRef.current = pbAnalyser;

      captureContextRef.current = new AudioContext({ sampleRate: 16000 });

      // RMS analyser (default sample rate)
      analyserContextRef.current = new AudioContext();
      const analyserSource = analyserContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = analyserContextRef.current.createAnalyser();
      analyserSource.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      const wsUrl = getVoiceWebSocketUrl(sessionId);
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setIsRecording(true);
        setIsConnecting(false);
        nextStartTimeRef.current = 0;

        // RMS animation — reads BOTH mic analyser and playback analyser
        const doRms = () => {
          // Mic RMS
          if (analyserRef.current) {
            const d = new Uint8Array(analyserRef.current.frequencyBinCount);
            analyserRef.current.getByteFrequencyData(d);
            setMicRms(d.reduce((a, b) => a + b) / d.length / 128);
          }
          // Playback (TTS) RMS
          if (playbackAnalyserRef.current) {
            const pd = new Uint8Array(playbackAnalyserRef.current.frequencyBinCount);
            playbackAnalyserRef.current.getByteFrequencyData(pd);
            const pbVal = pd.reduce((a, b) => a + b) / pd.length / 128;
            setPlaybackRms(pbVal);
          }
          animationFrameRef.current = requestAnimationFrame(doRms);
        };
        doRms();

        // Audio capture → send to gateway
        const source = captureContextRef.current!.createMediaStreamSource(stream);
        const processor = captureContextRef.current!.createScriptProcessor(2048, 1, 1);
        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const float32Data = e.inputBuffer.getChannelData(0);
          const int16 = new Int16Array(float32Data.length);
          for (let i = 0; i < float32Data.length; i++) {
            int16[i] = Math.max(-32768, Math.min(32767, float32Data[i] * 32768));
          }
          const frame = new Uint8Array(1 + int16.byteLength);
          frame[0] = 0x01;
          frame.set(new Uint8Array(int16.buffer), 1);
          ws.send(frame);
        };
        source.connect(processor);
        processor.connect(captureContextRef.current!.destination);
      };

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          try { handleGatewayRef.current(JSON.parse(event.data)); } catch {}
        } else if (event.data instanceof ArrayBuffer) {
          playPCM(event.data);
        }
      };

      ws.onclose = () => stopVoice();
      ws.onerror = () => stopVoice();
    } catch (err) {
      console.error("Failed to start voice:", err);
      setIsConnecting(false);
    }
  };

  const stopVoice = () => {
    setIsRecording(false);
    setIsConnecting(false);
    setMicRms(0);
    setPlaybackRms(0);
    setIsThinking(false);
    setIsSpeaking(false);
    setStreamingBotText("");
    setStreamingUserText("");
    streamingTextRef.current = "";
    streamingUserTextRef.current = "";
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (playbackTimerRef.current) cancelAnimationFrame(playbackTimerRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    try { captureContextRef.current?.close(); } catch {}
    try { playbackContextRef.current?.close(); } catch {}
    try { analyserContextRef.current?.close(); } catch {}
    if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); }
    wsRef.current = null;
    streamRef.current = null;
    captureContextRef.current = null;
    playbackContextRef.current = null;
    playbackAnalyserRef.current = null;
    analyserContextRef.current = null;
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || !sessionId) return;
    const text = inputText;
    setInputText("");
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      sendControl({ type: "inject_query", text });
    }
  };

  /* ═══════ EMPTY STATE ═══════ */
  if (!sessionId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center relative overflow-hidden">
        {/* Aurora background */}
        <div className="absolute inset-0 pointer-events-none aurora-bg" />
        {/* Neural grid */}
        <div className="absolute inset-0 pointer-events-none neural-grid opacity-[0.02]" />

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="relative z-10">
          {/* Floating 3D Brain Logo */}
          <div className="relative mb-10 inline-block">
            <div>
              <div className="relative w-28 h-28">
                {/* Hexagon frame */}
                <svg viewBox="0 0 200 200" className="absolute inset-0 w-full h-full">
                  <defs>
                    <linearGradient id="eHG" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#22d3ee" stopOpacity="0.2" /><stop offset="100%" stopColor="#a78bfa" stopOpacity="0.2" /></linearGradient>
                    <linearGradient id="eHS" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#22d3ee" stopOpacity="0.5" /><stop offset="100%" stopColor="#a78bfa" stopOpacity="0.5" /></linearGradient>
                  </defs>
                  <polygon points="100,10 180,50 180,150 100,190 20,150 20,50" fill="url(#eHG)" stroke="url(#eHS)" strokeWidth="1" />
                  <polygon points="100,35 155,60 155,140 100,165 45,140 45,60" fill="none" stroke="url(#eHS)" strokeWidth="0.3" opacity="0.2" />
                  {/* Brain network */}
                  <circle cx="100" cy="55" r="3" fill="#22d3ee" opacity="0.7" style={{ animation: 'brainPulse 2s ease-in-out infinite' }} />
                  <circle cx="70" cy="80" r="2.5" fill="#a78bfa" opacity="0.6" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.3s' }} />
                  <circle cx="130" cy="80" r="2.5" fill="#22d3ee" opacity="0.6" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.6s' }} />
                  <circle cx="100" cy="100" r="4" fill="white" opacity="0.8" style={{ animation: 'brainPulse 3s ease-in-out infinite' }} />
                  <circle cx="75" cy="120" r="2.5" fill="#22d3ee" opacity="0.6" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.9s' }} />
                  <circle cx="125" cy="120" r="2.5" fill="#a78bfa" opacity="0.6" style={{ animation: 'brainPulse 2s ease-in-out infinite 1.2s' }} />
                  <line x1="100" y1="55" x2="70" y2="80" stroke="#22d3ee" strokeWidth="0.7" opacity="0.35" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite' }} />
                  <line x1="100" y1="55" x2="130" y2="80" stroke="#a78bfa" strokeWidth="0.7" opacity="0.35" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 0.3s' }} />
                  <line x1="70" y1="80" x2="100" y2="100" stroke="#a78bfa" strokeWidth="0.5" opacity="0.25" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 0.6s' }} />
                  <line x1="130" y1="80" x2="100" y2="100" stroke="#22d3ee" strokeWidth="0.5" opacity="0.25" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 0.9s' }} />
                  <line x1="100" y1="100" x2="75" y2="120" stroke="#22d3ee" strokeWidth="0.5" opacity="0.25" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 1.2s' }} />
                  <line x1="100" y1="100" x2="125" y2="120" stroke="#a78bfa" strokeWidth="0.5" opacity="0.25" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 1.5s' }} />
                </svg>
                {/* AN text */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-3xl font-black font-display gradient-text-bright" style={{ filter: 'drop-shadow(0 0 12px rgba(6,182,212,0.25))' }}>AN</span>
                </div>
              </div>
            </div>
          </div>

          <h2 className="text-3xl md:text-4xl font-bold font-display gradient-text-bright mb-3 tracking-tight">Welcome to AskNova</h2>
          <p className="text-dark-200 max-w-md text-base leading-relaxed mb-8">Start a conversation to experience next-gen AI voice and chat.</p>
          <div className="flex items-center gap-2.5 justify-center flex-wrap">
            {[{ icon: <Brain size={13} />, label: "Neural AI", c: "#67e8f9" }, { icon: <Zap size={13} />, label: "Real-time", c: "#c4b5fd" }, { icon: <Sparkles size={13} />, label: "Intelligent", c: "#67e8f9" }].map((t, i) => (
              <motion.div key={i} whileHover={{ scale: 1.05, y: -2 }}
                className="flex items-center gap-1.5 px-3.5 py-2 rounded-full text-xs font-medium cursor-default"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', color: t.c }}>
                {t.icon} {t.label}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    );
  }

  /* ═══════ MAIN VIEW ═══════ */
  return (
    <div className="flex-1 flex flex-col overflow-hidden h-screen">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-5 shrink-0 z-20 circuit-glow"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(4,4,10,0.6)', backdropFilter: 'blur(20px)' }}>
        <div className="flex items-center gap-2.5">
          <BotAvatar />
          <div>
            <h2 className="text-sm font-bold text-white font-display leading-tight">AskNova</h2>
            <div className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /><p className="text-[9px] text-dark-300 font-medium">Neural Engine Active</p></div>
          </div>
        </div>
        <button onClick={() => setIsMuted(!isMuted)} className={`p-2 rounded-lg transition-all ${isMuted ? 'bg-red-500/10 text-red-400' : 'text-dark-300 hover:text-neon-400 hover:bg-white/[0.02]'}`}>
          {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>
      </div>

      {/* Grid layout */}
      <div className="flex-1 overflow-y-auto p-3 md:p-5">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-3 md:gap-5">
          <div className="lg:col-span-2 space-y-3 md:space-y-5">
            {/* Voice Orb */}
            <div className="glass-card rounded-2xl flex flex-col items-center justify-center relative overflow-hidden h-[320px] md:h-[380px]">
              {/* Neural dot grid background */}
              <div className="absolute inset-0 pointer-events-none neural-grid opacity-[0.03]" />
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[250px] h-[250px] rounded-full blur-[100px]" style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.04), transparent)' }} />
              </div>
              <div className="absolute top-4 left-5 flex items-center gap-1.5 z-10">
                <span className="w-1.5 h-1.5 rounded-full bg-neon-400 animate-pulse" />
                <span className="text-[9px] font-bold text-dark-400 uppercase tracking-[0.15em]">Neural Link</span>
              </div>
              <VoiceOrb isActive={isRecording} isConnecting={isConnecting} onToggle={isRecording ? stopVoice : startVoice} rms={rms} />
              <Waveform rms={rms} color="neon" />
            </div>

            {/* Conversation */}
            <div className="glass-card rounded-2xl flex flex-col overflow-hidden h-[380px] md:h-[460px]">
              <div className="px-5 py-3.5 flex items-center justify-between shrink-0" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                <div className="flex items-center gap-2 text-white font-semibold text-sm font-display">
                  <BotMessageSquare size={14} className="text-neon-400" />
                  Conversation
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-bold text-dark-400 uppercase tracking-wider">{messages.length} msg</span>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto px-4 py-4 md:px-5 space-y-4">
                <AnimatePresence>
                  {messages.map((msg, i) => (
                    <motion.div key={i}
                      initial={{ opacity: 0, y: 15, scale: 0.96 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1], delay: 0.05 }}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      {msg.role === "user" ? (
                        /* ── User bubble ── */
                        <div className="flex items-end gap-2 max-w-[80%]">
                          <motion.div
                            whileHover={{ scale: 1.01, y: -1 }}
                            className="ai-bubble-user rounded-2xl rounded-br-sm px-4 py-3 relative overflow-hidden"
                          >
                            {/* Shimmer sweep */}
                            <div className="absolute inset-0 shimmer pointer-events-none opacity-40" />
                            {/* Top edge glow */}
                            <div className="absolute top-0 left-0 right-0 h-[1px]"
                              style={{ background: 'linear-gradient(90deg, transparent, rgba(6,182,212,0.2), transparent)' }} />
                            {msg.bargeIn && (
                              <div className="flex items-center gap-1 mb-1.5 relative z-10">
                                <Zap size={10} className="text-amber-400" />
                                <span className="text-[9px] font-bold text-amber-400/80 uppercase tracking-wider">Barge-in</span>
                              </div>
                            )}
                            <p className="text-sm leading-relaxed text-white/90 relative z-10">{msg.content}</p>
                            {/* Timestamp */}
                            <p className="text-[9px] text-neon-400/30 mt-1.5 text-right relative z-10">You</p>
                          </motion.div>
                          <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
                            style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.12), rgba(139,92,246,0.12))', border: '1px solid rgba(6,182,212,0.1)' }}>
                            <User size={12} className="text-neon-300" />
                          </div>
                        </div>
                      ) : (
                        /* ── Bot bubble ── */
                        <div className="flex items-end gap-2.5 max-w-[85%]">
                          <BotAvatar />
                          <motion.div
                            whileHover={{ scale: 1.01, y: -1 }}
                            className="ai-bubble-bot rounded-2xl rounded-bl-sm px-4 py-3 relative overflow-hidden"
                          >
                            {/* Left gradient accent bar */}
                            <div className="absolute left-0 top-2 bottom-2 w-[2px] rounded-full"
                              style={{ background: msg.interrupted ? 'linear-gradient(to bottom, #f59e0b, #ef4444)' : 'linear-gradient(to bottom, #22d3ee, #a78bfa)' }} />
                            <p className="text-sm leading-relaxed text-dark-100 relative z-10 pl-2">{msg.content}{msg.interrupted && <span className="text-amber-400/60">…</span>}</p>
                            {/* AskNova label */}
                            <div className="flex items-center gap-1 mt-1.5 pl-2 relative z-10">
                              {msg.interrupted ? (
                                <>
                                  <Zap size={8} className="text-amber-400/50" />
                                  <p className="text-[9px] text-amber-400/40">Interrupted</p>
                                </>
                              ) : (
                                <>
                                  <Brain size={8} className="text-violet-400/40" />
                                  <p className="text-[9px] text-violet-400/30">AskNova</p>
                                </>
                              )}
                            </div>
                          </motion.div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>
                {/* Streaming user speech (word by word) */}
                {streamingUserText && (
                  <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="flex justify-end">
                    <div className="flex items-end gap-2 max-w-[80%]">
                      <motion.div className="ai-bubble-user rounded-2xl rounded-br-sm px-4 py-3 relative overflow-hidden">
                        <div className="absolute inset-0 shimmer pointer-events-none opacity-40" />
                        <div className="absolute top-0 left-0 right-0 h-[1px]" style={{ background: 'linear-gradient(90deg, transparent, rgba(6,182,212,0.2), transparent)' }} />
                        <p className="text-sm leading-relaxed text-white/90 relative z-10">{streamingUserText}<span className="inline-block w-0.5 h-4 ml-0.5 bg-neon-400/60 animate-pulse align-text-bottom" /></p>
                        <p className="text-[9px] text-neon-400/30 mt-1.5 text-right relative z-10">You</p>
                      </motion.div>
                      <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
                        style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.12), rgba(139,92,246,0.12))', border: '1px solid rgba(6,182,212,0.1)' }}>
                        <User size={12} className="text-neon-300" />
                      </div>
                    </div>
                  </motion.div>
                )}
                {/* Streaming bot response */}
                {streamingBotText && (
                  <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="flex justify-start">
                    <div className="flex items-end gap-2.5 max-w-[85%]">
                      <BotAvatar />
                      <div className="ai-bubble-bot rounded-2xl rounded-bl-sm px-4 py-3 relative overflow-hidden">
                        <div className="absolute left-0 top-2 bottom-2 w-[2px] rounded-full" style={{ background: 'linear-gradient(to bottom, #22d3ee, #a78bfa)' }} />
                        <p className="text-sm leading-relaxed text-dark-100 relative z-10 pl-2">{streamingBotText}</p>
                        <div className="flex items-center gap-1 mt-1.5 pl-2 relative z-10">
                          <Brain size={8} className="text-violet-400/40" />
                          <p className="text-[9px] text-violet-400/30">AskNova</p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
                {/* Typing indicator */}
                <AnimatePresence>{isThinking && !streamingBotText && <TypingIndicator />}</AnimatePresence>
                <div ref={messagesEndRef} />
              </div>
              {/* Input */}
              <div className="px-4 py-3 md:px-5 shrink-0" style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}>
                <form onSubmit={handleTextSubmit} className="relative">
                  <input type="text" placeholder="Ask Nova anything..."
                    className="w-full pl-4 pr-12 py-3 rounded-xl text-sm text-white outline-none transition-all placeholder:text-dark-400"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
                    onFocus={(e) => { e.target.style.borderColor = 'rgba(6,182,212,0.2)'; e.target.style.boxShadow = '0 0 0 3px rgba(6,182,212,0.05)'; }}
                    onBlur={(e) => { e.target.style.borderColor = 'rgba(255,255,255,0.06)'; e.target.style.boxShadow = 'none'; }}
                    value={inputText} onChange={(e) => setInputText(e.target.value)} />
                  <motion.button type="submit" whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg flex items-center justify-center text-white"
                    style={{ background: 'linear-gradient(135deg, #06b6d4, #7c3aed)', boxShadow: '0 2px 12px rgba(6,182,212,0.15)' }}>
                    <Send size={14} />
                  </motion.button>
                </form>
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div className="lg:col-span-1 space-y-3 md:space-y-5">
            <div className="glass-card rounded-2xl p-4">
              <div className="flex items-center gap-2 mb-3 text-white font-semibold text-sm font-display"><Settings size={14} className="text-violet-400" />Settings</div>
              <div className="space-y-2">
                {[{ label: "Voice Model", value: "Azure Neural", c: "neon" }, { label: "AI Engine", value: "Llama 3.2-3B", c: "violet" }].map((item, i) => (
                  <div key={i} className="flex items-center justify-between p-2.5 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                    <span className="text-xs text-dark-200">{item.label}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${item.c === 'neon' ? 'text-neon-400 bg-neon-400/10 border border-neon-400/15' : 'text-violet-400 bg-violet-400/10 border border-violet-400/15'}`}>{item.value}</span>
                  </div>
                ))}
                <div className="flex items-center justify-between p-2.5 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <span className="text-xs text-dark-200">Noise Suppression</span>
                  <div className="w-8 h-4 rounded-full relative cursor-pointer" style={{ background: 'linear-gradient(135deg, #06b6d4, #7c3aed)' }}><div className="absolute right-0.5 top-0.5 w-3 h-3 bg-white rounded-full shadow-sm" /></div>
                </div>
              </div>
            </div>
            <MetricsPanel gpuMetrics={gpuMetrics} latencyData={latencyData} />
            <div className="rounded-2xl p-5 relative overflow-hidden circuit-glow" style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.06), rgba(139,92,246,0.06))', border: '1px solid rgba(6,182,212,0.08)' }}>
              <div className="absolute -right-6 -top-6 w-24 h-24 rounded-full blur-3xl" style={{ background: 'rgba(6,182,212,0.08)' }} />
              <div className="flex items-center gap-1.5 mb-2 relative z-10"><Brain size={13} className="text-neon-400" /><h4 className="text-xs font-bold font-display text-white">Neural Tip</h4></div>
              <p className="text-dark-200 text-[11px] leading-relaxed relative z-10">Interrupt the AI anytime by speaking. The neural pipeline resets automatically for seamless conversation.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
