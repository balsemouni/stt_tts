import React, { useMemo, useEffect, useRef, useState } from 'react';

interface WaveformProps {
  rms: number;
  color: string;
}

export const Waveform: React.FC<WaveformProps> = ({ rms }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);
  const smoothRmsRef = useRef(0);
  const [, setTick] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const draw = () => {
      timeRef.current += 0.025;
      smoothRmsRef.current += (rms - smoothRmsRef.current) * 0.12;
      const sr = smoothRmsRef.current;

      ctx.clearRect(0, 0, w, h);
      const mid = h / 2;

      // Draw 3 layered waves
      const waves = [
        { amp: 12 + sr * 35, freq: 0.012, speed: 1.2, color: 'rgba(6,182,212,', baseOp: 0.5, glowOp: 0.15 },
        { amp: 8 + sr * 25, freq: 0.018, speed: -0.8, color: 'rgba(139,92,246,', baseOp: 0.4, glowOp: 0.1 },
        { amp: 5 + sr * 15, freq: 0.025, speed: 1.5, color: 'rgba(103,232,249,', baseOp: 0.25, glowOp: 0.08 },
      ];

      for (const wave of waves) {
        // Glow pass
        ctx.beginPath();
        ctx.moveTo(0, mid);
        for (let x = 0; x <= w; x += 2) {
          const nx = x / w;
          const envelope = Math.sin(nx * Math.PI);
          const y = mid + Math.sin(x * wave.freq + timeRef.current * wave.speed) * wave.amp * envelope
            + Math.sin(x * wave.freq * 1.7 + timeRef.current * wave.speed * 0.6) * wave.amp * 0.3 * envelope;
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `${wave.color}${wave.glowOp})`;
        ctx.lineWidth = 6;
        ctx.filter = 'blur(4px)';
        ctx.stroke();
        ctx.filter = 'none';

        // Main line
        ctx.beginPath();
        ctx.moveTo(0, mid);
        for (let x = 0; x <= w; x += 2) {
          const nx = x / w;
          const envelope = Math.sin(nx * Math.PI);
          const y = mid + Math.sin(x * wave.freq + timeRef.current * wave.speed) * wave.amp * envelope
            + Math.sin(x * wave.freq * 1.7 + timeRef.current * wave.speed * 0.6) * wave.amp * 0.3 * envelope;
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `${wave.color}${wave.baseOp + sr * 0.3})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Center line gradient when idle
      if (sr < 0.05) {
        const grad = ctx.createLinearGradient(0, 0, w, 0);
        grad.addColorStop(0, 'rgba(6,182,212,0)');
        grad.addColorStop(0.3, 'rgba(6,182,212,0.1)');
        grad.addColorStop(0.5, 'rgba(139,92,246,0.15)');
        grad.addColorStop(0.7, 'rgba(6,182,212,0.1)');
        grad.addColorStop(1, 'rgba(6,182,212,0)');
        ctx.beginPath();
        ctx.moveTo(0, mid);
        ctx.lineTo(w, mid);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [rms]);

  return (
    <div className="absolute bottom-0 left-0 right-0 h-20 px-4">
      <canvas ref={canvasRef} className="w-full h-full" style={{ opacity: 0.9 }} />
    </div>
  );
};
