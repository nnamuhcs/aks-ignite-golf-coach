import { useState } from 'react'

const NODES = [
  { id: 'upload', label: 'Video Upload', detail: 'Athlete and optional reference', x: 420, y: 24, w: 210, type: 'frontend' },
  { id: 'api', label: 'FastAPI Analysis API', detail: 'Jobs, results, assets, telemetry', x: 420, y: 118, w: 210, type: 'api' },
  { id: 'decode', label: 'Video Decoder', detail: 'OpenCV frame sampling', x: 70, y: 240, w: 190, type: 'pipeline' },
  { id: 'pose', label: '3D Pose Estimation', detail: '33 body landmarks per frame', x: 305, y: 240, w: 210, type: 'model' },
  { id: 'phase', label: 'Swing Phase Detection', detail: 'Eight data driven phases', x: 560, y: 240, w: 210, type: 'pipeline' },
  { id: 'metrics', label: 'Biomechanics Engine', detail: 'Angles, tempo, trajectory', x: 795, y: 240, w: 190, type: 'pipeline' },
  { id: 'quality', label: 'Movement Quality', detail: 'Posture, extension and consistency', x: 170, y: 376, w: 210, type: 'model' },
  { id: 'coach', label: 'Coaching Engine', detail: 'Evidence based guidance', x: 420, y: 376, w: 210, type: 'pipeline' },
  { id: 'replay', label: 'Synchronized 3D Replay', detail: 'Source video and digital athlete', x: 670, y: 376, w: 220, type: 'frontend' },
  { id: 'platform', label: 'AKS Flex Node Telemetry', detail: 'Node, GPU, CPU, memory', x: 420, y: 500, w: 220, type: 'api' },
]

const EDGES = [
  ['upload', 'api'], ['api', 'decode'], ['decode', 'pose'], ['pose', 'phase'],
  ['phase', 'metrics'], ['metrics', 'quality'], ['metrics', 'coach'],
  ['quality', 'coach'], ['coach', 'replay'], ['api', 'platform'], ['platform', 'replay'],
]

const COLORS = {
  frontend: { fill: '#dbeafe', stroke: '#3b82f6' },
  api: { fill: '#ede9fe', stroke: '#8b5cf6' },
  pipeline: { fill: '#d1fae5', stroke: '#10b981' },
  model: { fill: '#fef3c7', stroke: '#f59e0b' },
}

function center(node) {
  return { x: node.x + node.w / 2, y: node.y + 30 }
}

export default function ArchitecturePanel({ kubernetesView }) {
  const [fullscreen, setFullscreen] = useState(false)
  const [highlighted, setHighlighted] = useState(null)
  const [activeTab, setActiveTab] = useState('architecture')

  if (activeTab === 'kubernetes') {
    return (
      <div className="arch-page kubernetes-mode">
        <div className="arch-header">
          <div>
            <h2>AKS Kubernetes Operations</h2>
            <p>Live workloads, services, node resources, and accelerator status</p>
          </div>
          <div className="architecture-tabs">
            <button onClick={() => setActiveTab('architecture')}>Architecture</button>
            <button className="active" onClick={() => setActiveTab('kubernetes')}>Kubernetes</button>
          </div>
        </div>
        <div className="architecture-tab-content">{kubernetesView}</div>
      </div>
    )
  }

  return (
    <div className="arch-page">
      <div className="arch-header">
        <div>
          <h2>System Architecture</h2>
          <p>Real 3D golf analysis on AKS Flex Node and NVIDIA Jetson Thor</p>
        </div>
        <div className="architecture-header-actions">
          <div className="architecture-tabs">
            <button className="active" onClick={() => setActiveTab('architecture')}>Architecture</button>
            <button onClick={() => setActiveTab('kubernetes')}>Kubernetes</button>
          </div>
          <button className="secondary-button" onClick={() => setFullscreen(!fullscreen)}>
            {fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          </button>
        </div>
      </div>

      <div className={`architecture-canvas ${fullscreen ? 'fullscreen' : ''}`}>
        {fullscreen && (
          <div className="architecture-toolbar">
            <strong>AKS Ignite 3D Golf Coach</strong>
            <button className="secondary-button" onClick={() => setFullscreen(false)}>Close</button>
          </div>
        )}
        <svg viewBox="0 0 1060 590" role="img" aria-label="Application architecture diagram">
          <defs>
            <pattern id="arch-dots" width="24" height="24" patternUnits="userSpaceOnUse">
              <circle cx="12" cy="12" r="1" fill="#dce5ea" />
            </pattern>
            <marker id="arch-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0 0 L8 4 L0 8Z" fill="#94a3b8" />
            </marker>
          </defs>
          <rect width="1060" height="590" fill="#f8fafc" />
          <rect width="1060" height="590" fill="url(#arch-dots)" />
          {EDGES.map(([fromId, toId]) => {
            const from = NODES.find((node) => node.id === fromId)
            const to = NODES.find((node) => node.id === toId)
            const start = center(from)
            const end = center(to)
            const active = highlighted && [fromId, toId].includes(highlighted)
            return (
              <path
                key={`${fromId}-${toId}`}
                d={`M${start.x},${start.y} C${start.x},${(start.y + end.y) / 2} ${end.x},${(start.y + end.y) / 2} ${end.x},${end.y}`}
                fill="none"
                stroke={active ? '#059669' : '#cbd5e1'}
                strokeWidth={active ? 2.5 : 1.5}
                markerEnd="url(#arch-arrow)"
                opacity={highlighted && !active ? 0.2 : 1}
              />
            )
          })}
          {NODES.map((node) => {
            const color = COLORS[node.type]
            return (
              <g
                key={node.id}
                onMouseEnter={() => setHighlighted(node.id)}
                onMouseLeave={() => setHighlighted(null)}
                className="architecture-node"
              >
                <rect
                  x={node.x}
                  y={node.y}
                  width={node.w}
                  height="60"
                  rx="10"
                  fill={color.fill}
                  stroke={highlighted === node.id ? color.stroke : '#d5dee5'}
                  strokeWidth={highlighted === node.id ? 2.5 : 1}
                />
                <text x={node.x + node.w / 2} y={node.y + 25} textAnchor="middle" className="architecture-node-title">
                  {node.label}
                </text>
                <text x={node.x + node.w / 2} y={node.y + 43} textAnchor="middle" className="architecture-node-detail">
                  {node.detail}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      <div className="architecture-details">
        <section>
          <h3>Kubernetes deployment</h3>
          <div className="architecture-tags">
            <span>backend Deployment</span><span>frontend Deployment</span><span>ClusterIP Service</span>
            <span>LoadBalancer Service</span><span>ServiceAccount</span><span>ClusterRole</span>
          </div>
        </section>
        <section>
          <h3>Technology</h3>
          <div className="architecture-tags">
            <span>React</span><span>Three.js</span><span>FastAPI</span><span>3D Pose Inference</span>
            <span>OpenCV</span><span>Kubernetes API</span><span>NVIDIA SMI</span>
          </div>
        </section>
        <section>
          <h3>Thor target</h3>
          <div className="architecture-tags">
            <span>ARM64</span><span>AKS Flex Node</span><span>NVIDIA RuntimeClass</span>
            <span>GPU device plugin</span><span>caldova-gpu-la</span>
          </div>
        </section>
      </div>
    </div>
  )
}
