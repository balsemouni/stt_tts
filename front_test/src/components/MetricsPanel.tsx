import React from 'react';
import { Cpu, Zap, Thermometer, Activity } from 'lucide-react';
import type { GPUData, LatencyData } from '../types';

interface MetricsPanelProps {
  gpuMetrics: GPUData;
  latencyData: LatencyData;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ gpuMetrics, latencyData }) => {
  return (
    <div className="space-y-4">
      {/* GPU Metrics */}
      <div className="glass-card rounded-3xl p-5">
        <div className="flex items-center gap-2 mb-4 text-white font-bold text-sm font-display">
          <Cpu size={16} className="text-neon-400" />
          System Performance
        </div>
        <div className="grid grid-cols-2 gap-3">
          <MetricItem label="Utilization" value={`${gpuMetrics.utilization.toFixed(1)}%`} icon={<Activity size={12} />} color="neon" />
          <MetricItem label="Memory" value={`${gpuMetrics.memory.toFixed(1)} GB`} icon={<Zap size={12} />} color="violet" />
          <MetricItem label="Temp" value={`${gpuMetrics.temperature.toFixed(1)}°C`} icon={<Thermometer size={12} />} color="neon" />
          <MetricItem label="Power" value={`${gpuMetrics.power.toFixed(0)}W`} icon={<Zap size={12} />} color="violet" />
        </div>
      </div>

      {/* Latency Metrics */}
      <div className="glass-card rounded-3xl p-5">
        <div className="flex items-center gap-2 mb-4 text-white font-bold text-sm font-display">
          <Activity size={16} className="text-violet-400" />
          Pipeline Latency
        </div>
        <div className="space-y-3">
          <LatencyItem label="STT (Speech-to-Text)" value={`${latencyData.avg_stt.toFixed(0)}ms`} progress={(latencyData.avg_stt / 300) * 100} color="neon" />
          <LatencyItem label="LLM (Reasoning)" value={`${latencyData.avg_llm.toFixed(0)}ms`} progress={(latencyData.avg_llm / 600) * 100} color="violet" />
          <LatencyItem label="TTS (Text-to-Speech)" value={`${latencyData.avg_tts.toFixed(0)}ms`} progress={(latencyData.avg_tts / 350) * 100} color="neon" />
        </div>
      </div>
    </div>
  );
};

const MetricItem = ({ label, value, icon, color }: { label: string, value: string, icon: React.ReactNode, color: 'neon' | 'violet' }) => (
  <div className="bg-white/[0.03] rounded-xl p-3 border border-white/[0.05] transition-all hover:bg-white/[0.05]">
    <div className={`flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider mb-1 ${
      color === 'neon' ? 'text-neon-400/60' : 'text-violet-400/60'
    }`}>
      {icon}
      {label}
    </div>
    <div className="text-white font-bold text-sm">{value}</div>
  </div>
);

const LatencyItem = ({ label, value, progress, color }: { label: string, value: string, progress: number, color: 'neon' | 'violet' }) => (
  <div>
    <div className="flex justify-between text-[11px] mb-1.5">
      <span className="text-dark-200 font-medium">{label}</span>
      <span className={`font-bold ${color === 'neon' ? 'text-neon-400' : 'text-violet-400'}`}>{value}</span>
    </div>
    <div className="h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
      <div 
        className="h-full rounded-full transition-all duration-500"
        style={{ 
          width: `${Math.min(100, progress)}%`,
          background: color === 'neon' 
            ? 'linear-gradient(90deg, #06b6d4, #22d3ee)' 
            : 'linear-gradient(90deg, #7c3aed, #a78bfa)',
          boxShadow: color === 'neon' 
            ? '0 0 8px rgba(6,182,212,0.4)' 
            : '0 0 8px rgba(139,92,246,0.4)',
        }}
      />
    </div>
  </div>
);
