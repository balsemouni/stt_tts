import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";

interface SplashScreenProps {
  onComplete: () => void;
  isLoggedIn?: boolean;
}

/* ── AI Brain SVG ────────────────────────────────────────────────── */
const AIBrainSVG = ({ phase }: { phase: number }) => (
  <svg viewBox="0 0 200 200" className="absolute inset-0 w-full h-full" style={{ filter: 'drop-shadow(0 0 12px rgba(6,182,212,0.3))' }}>
    <defs>
      <linearGradient id="brainGradCyan" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.9" />
        <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
      </linearGradient>
      <linearGradient id="brainGradViolet" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.9" />
        <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.4" />
      </linearGradient>
      <filter id="nodeGlow">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
    </defs>
    {[
      { x1: 100, y1: 52, x2: 65, y2: 78, c: 'url(#brainGradCyan)' },
      { x1: 100, y1: 52, x2: 135, y2: 78, c: 'url(#brainGradViolet)' },
      { x1: 65, y1: 78, x2: 45, y2: 110, c: 'url(#brainGradViolet)' },
      { x1: 135, y1: 78, x2: 155, y2: 110, c: 'url(#brainGradCyan)' },
      { x1: 65, y1: 78, x2: 100, y2: 110, c: 'url(#brainGradCyan)' },
      { x1: 135, y1: 78, x2: 100, y2: 110, c: 'url(#brainGradViolet)' },
      { x1: 45, y1: 110, x2: 75, y2: 142, c: 'url(#brainGradCyan)' },
      { x1: 155, y1: 110, x2: 125, y2: 142, c: 'url(#brainGradViolet)' },
      { x1: 100, y1: 110, x2: 75, y2: 142, c: 'url(#brainGradViolet)' },
      { x1: 100, y1: 110, x2: 125, y2: 142, c: 'url(#brainGradCyan)' },
    ].map((line, i) => (
      <line key={i} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2}
        stroke={line.c} strokeWidth="1.2" opacity={phase >= 0 ? 0.6 : 0}
        strokeDasharray="4 4"
        style={{ animation: phase >= 0 ? `dataFlow 2s linear infinite ${i * 0.2}s` : 'none' }}
      />
    ))}
    {[
      { cx: 100, cy: 52, r: 5, c: '#22d3ee', d: 0 },
      { cx: 65, cy: 78, r: 4, c: '#a78bfa', d: 0.1 },
      { cx: 135, cy: 78, r: 4, c: '#22d3ee', d: 0.2 },
      { cx: 45, cy: 110, r: 3.5, c: '#a78bfa', d: 0.3 },
      { cx: 100, cy: 110, r: 6, c: '#ffffff', d: 0.15 },
      { cx: 155, cy: 110, r: 3.5, c: '#22d3ee', d: 0.35 },
      { cx: 75, cy: 142, r: 4, c: '#22d3ee', d: 0.4 },
      { cx: 125, cy: 142, r: 4, c: '#a78bfa', d: 0.45 },
    ].map((node, i) => (
      <g key={i} filter="url(#nodeGlow)">
        <circle cx={node.cx} cy={node.cy} r={node.r + 4} fill="none" stroke={node.c} strokeWidth="0.5"
          opacity={phase >= 0 ? 0.2 : 0}
          style={{ animation: phase >= 0 ? `brainPulse 2s ease-in-out infinite ${node.d}s` : 'none' }} />
        <circle cx={node.cx} cy={node.cy} r={node.r} fill={node.c}
          opacity={phase >= 0 ? 0.9 : 0}
          style={{ animation: phase >= 0 ? `brainPulse 2s ease-in-out infinite ${node.d}s` : 'none' }} />
      </g>
    ))}
    <circle cx="100" cy="100" r="12" fill="none" stroke="#22d3ee" strokeWidth="0.5" opacity="0.2"
      style={{ animation: 'brainPulse 3s ease-in-out infinite' }} />
    <circle cx="100" cy="100" r="18" fill="none" stroke="#a78bfa" strokeWidth="0.3" opacity="0.15"
      style={{ animation: 'brainPulse 4s ease-in-out infinite 0.5s' }} />
  </svg>
);

export const SplashScreen: React.FC<SplashScreenProps> = ({ onComplete, isLoggedIn = false }) => {
  // 0=logo build, 1=name reveal, 2=tagline+bar, 3=zoom-out transition, 4=done
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase(1), 900),
      setTimeout(() => setPhase(2), 2000),
      setTimeout(() => setPhase(3), 3200),   // start zoom-out
      setTimeout(() => setPhase(4), 4400),   // zoom-out complete
      setTimeout(() => onComplete(), 4600),  // remove from DOM
    ];
    return () => timers.forEach(clearTimeout);
  }, [onComplete]);

  // During phase 3+, the logo shrinks and moves up to where the Auth page logo sits
  // The background fades to transparent, revealing the Auth page underneath
  const isZooming = phase >= 3;
  const isDone = phase >= 4;

  return (
    <AnimatePresence>
      {!isDone && (
        <motion.div
          className="fixed inset-0 z-[9999] flex items-center justify-center overflow-hidden"
          style={{ background: '#030308', pointerEvents: isZooming ? 'none' : 'auto' }}
          animate={{
            backgroundColor: isZooming ? 'rgba(3,3,8,0)' : 'rgba(3,3,8,1)',
          }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1.2, ease: [0.4, 0, 0.2, 1] }}
        >
          {/* Animated grid floor — fades out during zoom */}
          <motion.div
            className="absolute inset-0 pointer-events-none overflow-hidden"
            animate={{ opacity: isZooming ? 0 : 1 }}
            transition={{ duration: 0.8 }}
          >
            <div className="absolute bottom-0 left-[-50%] right-[-50%] h-[70vh] opacity-[0.06]"
              style={{
                backgroundImage: 'linear-gradient(rgba(139,92,246,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.5) 1px, transparent 1px)',
                backgroundSize: '80px 80px',
                transform: 'perspective(400px) rotateX(65deg)',
                transformOrigin: 'bottom center',
                animation: 'gridScroll 4s linear infinite',
              }}
            />
          </motion.div>

          {/* Neural dot grid — fades out */}
          <motion.div
            className="absolute inset-0 neural-grid opacity-[0.03] pointer-events-none"
            animate={{ opacity: isZooming ? 0 : 0.03 }}
            transition={{ duration: 0.6 }}
          />

          {/* Aurora glow layers — fade out */}
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{
              scale: isZooming ? 0.5 : [0, 2, 1.5],
              opacity: isZooming ? 0 : [0, 0.25, 0.1],
            }}
            transition={{ duration: isZooming ? 0.8 : 2.5, ease: "easeOut" }}
            className="absolute w-[700px] h-[700px] rounded-full"
            style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.15), rgba(6,182,212,0.08), transparent 65%)' }}
          />

          {/* DNA helix — fades out */}
          <motion.div
            className="absolute inset-0 pointer-events-none"
            animate={{ opacity: isZooming ? 0 : 1 }}
            transition={{ duration: 0.6 }}
          >
            {[...Array(16)].map((_, i) => {
              const angle = (i / 16) * Math.PI * 2;
              const radius = 180;
              const x = Math.cos(angle) * radius;
              const y = Math.sin(angle) * radius * 0.3;
              const isBack = Math.sin(angle) < 0;
              return (
                <motion.div key={`helix-${i}`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: isBack ? 0.15 : 0.5, rotate: 360 }}
                  transition={{ rotate: { duration: 12, repeat: Infinity, ease: "linear" }, opacity: { duration: 1, delay: i * 0.05 } }}
                  className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
                  style={{ width: '360px', height: '360px', transformStyle: 'preserve-3d' }}
                >
                  <div className="absolute rounded-full"
                    style={{
                      width: `${3 + (isBack ? 0 : 1)}px`,
                      height: `${3 + (isBack ? 0 : 1)}px`,
                      left: `calc(50% + ${x}px)`,
                      top: `calc(50% + ${y}px)`,
                      background: i % 2 === 0 ? '#22d3ee' : '#a78bfa',
                      boxShadow: `0 0 ${isBack ? 3 : 8}px ${i % 2 === 0 ? 'rgba(34,211,238,0.6)' : 'rgba(167,139,250,0.6)'}`,
                      zIndex: isBack ? 0 : 2,
                    }}
                  />
                </motion.div>
              );
            })}
            {[...Array(8)].map((_, i) => (
              <motion.div key={`orbit2-${i}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 0.4, 0], rotate: -360 }}
                transition={{ rotate: { duration: 18, repeat: Infinity, ease: "linear" }, opacity: { duration: 3, repeat: Infinity, delay: i * 0.4 } }}
                className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
                style={{ width: `${260 + i * 15}px`, height: `${260 + i * 15}px` }}
              >
                <div className="absolute rounded-full"
                  style={{
                    width: '2px', height: '2px', top: 0, left: '50%',
                    background: i % 2 === 0 ? '#67e8f9' : '#c4b5fd',
                    boxShadow: `0 0 6px ${i % 2 === 0 ? 'rgba(103,232,249,0.5)' : 'rgba(196,181,253,0.5)'}`,
                  }}
                />
              </motion.div>
            ))}
          </motion.div>

          {/* ═══════ MAIN CONTENT — logo + text ═══════ */}
          {/* Zooms to Auth logo (center-top) or Sidebar logo (top-left) */}
          <motion.div
            className="relative z-10 flex flex-col items-center"
            style={{ perspective: '1200px' }}
            animate={{
              x: isZooming ? (isLoggedIn ? -(window.innerWidth / 2 - 50) : 0) : 0,
              y: isZooming ? (isLoggedIn ? -(window.innerHeight / 2 - 40) : -180) : 0,
              scale: isZooming ? (isLoggedIn ? 0.2 : 0.35) : 1,
            }}
            transition={{
              duration: 1.2,
              ease: [0.4, 0, 0.2, 1],
            }}
          >
            {/* 3D Logo with AI brain inside */}
            <motion.div
              initial={{ rotateY: -180, rotateX: 20, opacity: 0, scale: 0.2 }}
              animate={{
                rotateY: 0,
                rotateX: 0,
                opacity: isZooming ? 0.9 : 1,
                scale: 1,
              }}
              transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
              style={{ transformStyle: 'preserve-3d' }}
            >
              <motion.div
                animate={isZooming
                  ? { rotateY: 0, rotateX: 0 }
                  : { rotateY: [0, 6, -6, 0], rotateX: [0, -3, 3, 0] }
                }
                transition={{ duration: isZooming ? 0.5 : 8, repeat: isZooming ? 0 : Infinity, ease: "easeInOut" }}
                className="w-40 h-40 md:w-48 md:h-48 relative"
                style={{ transformStyle: 'preserve-3d' }}
              >
                {/* Hexagonal frame */}
                <svg viewBox="0 0 200 200" className="absolute inset-0 w-full h-full">
                  <defs>
                    <linearGradient id="hexGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.2" />
                      <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.12" />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.2" />
                    </linearGradient>
                    <linearGradient id="hexStroke" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.7" />
                      <stop offset="50%" stopColor="#a78bfa" stopOpacity="0.7" />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.7" />
                    </linearGradient>
                    <filter id="hexGlow">
                      <feGaussianBlur stdDeviation="5" result="blur" />
                      <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                    </filter>
                  </defs>
                  <polygon points="100,10 180,50 180,150 100,190 20,150 20,50"
                    fill="url(#hexGrad)" stroke="url(#hexStroke)" strokeWidth="1.5" filter="url(#hexGlow)" />
                  <polygon points="100,35 155,60 155,140 100,165 45,140 45,60"
                    fill="none" stroke="url(#hexStroke)" strokeWidth="0.5" opacity="0.25" />
                  <polygon points="100,55 135,72 135,128 100,145 65,128 65,72"
                    fill="none" stroke="url(#hexStroke)" strokeWidth="0.3" opacity="0.15" />
                </svg>

                {/* AI Brain neural network */}
                <AIBrainSVG phase={phase} />

                {/* "AN" text */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-5xl md:text-6xl font-black font-display tracking-tighter"
                    style={{
                      background: 'linear-gradient(135deg, #22d3ee, #ffffff, #a78bfa)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      filter: 'drop-shadow(0 0 25px rgba(6,182,212,0.35))',
                    }}>AN</span>
                </div>

                {/* Rotating outer ring — stops during zoom */}
                <motion.div
                  animate={{ rotate: isZooming ? 0 : 360 }}
                  transition={{ duration: 20, repeat: isZooming ? 0 : Infinity, ease: "linear" }}
                  className="absolute -inset-5"
                >
                  <svg viewBox="0 0 220 220" className="w-full h-full">
                    <circle cx="110" cy="110" r="105" fill="none" stroke="url(#hexStroke)" strokeWidth="0.5"
                      strokeDasharray="8 16" opacity="0.3" />
                  </svg>
                </motion.div>

                {/* Pulse rings — fade during zoom */}
                <motion.div
                  animate={{
                    scale: isZooming ? 1 : [1, 1.4, 1],
                    opacity: isZooming ? 0 : [0.3, 0, 0.3],
                  }}
                  transition={{ duration: isZooming ? 0.5 : 2.5, repeat: isZooming ? 0 : Infinity }}
                  className="absolute -inset-6 rounded-full border border-neon-400/25"
                />
                <motion.div
                  animate={{
                    scale: isZooming ? 1 : [1, 1.6, 1],
                    opacity: isZooming ? 0 : [0.15, 0, 0.15],
                  }}
                  transition={{ duration: isZooming ? 0.5 : 3.5, repeat: isZooming ? 0 : Infinity, delay: isZooming ? 0 : 0.5 }}
                  className="absolute -inset-10 rounded-full border border-violet-400/15"
                />
              </motion.div>
            </motion.div>

            {/* Company name — fades out during zoom */}
            <motion.div
              initial={{ opacity: 0, y: 35, filter: 'blur(10px)' }}
              animate={{
                opacity: isZooming ? 0 : (phase >= 1 ? 1 : 0),
                y: isZooming ? -20 : (phase >= 1 ? 0 : 35),
                filter: phase >= 1 ? 'blur(0px)' : 'blur(10px)',
              }}
              transition={{ duration: isZooming ? 0.6 : 1, ease: [0.16, 1, 0.3, 1] }}
              className="mt-10 text-center"
            >
              <h1 className="text-5xl md:text-7xl font-black font-display tracking-tight"
                style={{
                  background: 'linear-gradient(135deg, #22d3ee 0%, #ffffff 35%, #a78bfa 65%, #22d3ee 100%)',
                  backgroundSize: '200% 100%',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  animation: phase >= 1 && !isZooming ? 'textShimmer 4s ease-in-out infinite' : 'none',
                  filter: 'drop-shadow(0 0 40px rgba(6,182,212,0.25))',
                }}>AskNova</h1>
            </motion.div>

            {/* Tagline — fades out during zoom */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{
                opacity: isZooming ? 0 : (phase >= 2 ? 1 : 0),
                y: isZooming ? -10 : (phase >= 2 ? 0 : 20),
              }}
              transition={{ duration: isZooming ? 0.4 : 0.7, ease: "easeOut" }}
              className="mt-5 flex items-center gap-4"
            >
              <div className="h-[1px] w-16 bg-gradient-to-r from-transparent to-neon-400/50" />
              <span className="text-dark-200 text-sm md:text-base font-medium tracking-[0.3em] uppercase">
                Intelligence Redefined
              </span>
              <div className="h-[1px] w-16 bg-gradient-to-l from-transparent to-violet-400/50" />
            </motion.div>

            {/* Loading bar — fades out during zoom */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: isZooming ? 0 : (phase >= 2 ? 1 : 0) }}
              transition={{ delay: isZooming ? 0 : 0.3, duration: 0.4 }}
              className="mt-8 w-56 h-[3px] bg-white/5 rounded-full overflow-hidden"
            >
              <motion.div
                initial={{ width: '0%' }}
                animate={{ width: phase >= 2 ? '100%' : '0%' }}
                transition={{ duration: 1.5, ease: "easeInOut" }}
                className="h-full rounded-full"
                style={{ background: 'linear-gradient(90deg, #22d3ee, #8b5cf6, #22d3ee)', boxShadow: '0 0 12px rgba(6,182,212,0.4)' }}
              />
            </motion.div>
          </motion.div>

          <style>{`
            @keyframes gridScroll {
              0% { background-position: 0 0; }
              100% { background-position: 0 80px; }
            }
            @keyframes textShimmer {
              0%, 100% { background-position: 0% 50%; }
              50% { background-position: 100% 50%; }
            }
          `}</style>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
