export interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string | null;
  is_active?: boolean;
  roles?: string[];
}

export interface Session {
  id: string;
  user_id: string;
  title: string | null;
  created_at: string;
  updated_at?: string;
}

export interface Message {
  id?: string;
  session_id?: string;
  role: string;
  content: string;
  created_at?: string;
  bargeIn?: boolean;
  interrupted?: boolean;
}

export interface GPUData {
  utilization: number;
  memory: number;
  total_memory: number;
  memory_percent: number;
  temperature: number;
  power: number;
  power_limit: number;
}

export interface LatencyData {
  avg_stt: number;
  avg_llm: number;
  avg_tts: number;
  p95_stt: number;
  p95_llm: number;
  p95_tts: number;
}

export interface ChatResponse {
  reply: string;
  session_id?: string;
}

export interface ApiError {
  error: string;
  status?: number;
}
