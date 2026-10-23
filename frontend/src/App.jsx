import { useEffect, useRef, useState } from 'react'
import ArchitecturePanel from './ArchitecturePanel'
import ThreePose from './ThreePose'

const API_BASE = import.meta.env.VITE_API_BASE || ''

function titleCase(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function gpuMemoryLabel(gpu) {
  return gpu.memory_used_mb != null && gpu.memory_total_mb != null
    ? `${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB GPU memory`
    : 'GPU memory not reported by NVIDIA'
}

function K8sPanel({ isOpen, onToggle, platform, result }) {
  const memoryPercent = platform
    ? Math.round((platform.memory_used_mb / platform.memory_total_mb) * 100)
    : 0
  const gpuPercent = platform?.gpu.utilization_percent
  const kubernetes = platform?.kubernetes

  return (
    <aside className={`k8s-wrapper ${isOpen ? 'open' : ''}`}>
      <button className="k8s-toggle" onClick={onToggle}>
        <span className="k8s-symbol">K8s</span>
        <span>{isOpen ? 'Close' : 'Live'}</span>
      </button>
      <div className="k8s-panel">
        <div className="k8s-panel-header">
          <div>
            <strong>Kubernetes and Compute</strong>
            <small>{kubernetes?.node_name || platform?.hostname || 'Connecting'}</small>
          </div>
          <span className={`environment-badge ${kubernetes?.in_cluster ? 'aks' : 'local'}`}>
            {kubernetes?.in_cluster ? 'AKS Flex Node' : 'Local'}
          </span>
        </div>

        <section className="k8s-section">
          <h3>Components</h3>
          <div className="component-list">
            <div className="component-row">
              <i className="status-ready" />
              <div><strong>golf-coach-frontend</strong><small>React and Three.js</small></div>
              <b>Ready</b>
            </div>
            <div className="component-row">
              <i className={platform ? 'status-ready' : 'status-waiting'} />
              <div><strong>golf-coach-backend</strong><small>FastAPI analysis service</small></div>
              <b>{platform ? 'Ready' : 'Waiting'}</b>
            </div>
            <div className="component-row">
              <i className={platform?.gpu.available ? 'status-ready' : 'status-off'} />
              <div><strong>NVIDIA GPU</strong><small>{platform?.gpu.name || 'Not detected on this host'}</small></div>
              <b>{platform?.gpu.available ? 'Live' : 'Off'}</b>
            </div>
          </div>
        </section>

        <section className="k8s-section">
          <h3>Live system metrics</h3>
          <div className="system-metric">
            <div><span>GPU utilization</span><strong>{platform?.gpu.available && gpuPercent != null ? `${gpuPercent}%` : 'N/A'}</strong></div>
            <div className="metric-track"><i className="gpu-fill" style={{ width: `${gpuPercent ?? 0}%` }} /></div>
            <small>
              {platform?.gpu.available
                ? gpuMemoryLabel(platform.gpu)
                : 'Live NVIDIA telemetry appears on Jetson Thor'}
            </small>
          </div>
          <div className="system-metric">
            <div><span>CPU utilization</span><strong>{platform ? `${platform.cpu_utilization_percent.toFixed(1)}%` : 'N/A'}</strong></div>
            <div className="metric-track"><i style={{ width: `${platform?.cpu_utilization_percent || 0}%` }} /></div>
            <small>{platform?.cpu_count || 0} logical cores</small>
          </div>
          <div className="system-metric">
            <div><span>Memory</span><strong>{platform ? `${memoryPercent}%` : 'N/A'}</strong></div>
            <div className="metric-track"><i style={{ width: `${memoryPercent}%` }} /></div>
            <small>
              {platform
                ? `${Math.round(platform.memory_used_mb / 1024)} / ${Math.round(platform.memory_total_mb / 1024)} GB`
                : 'Reading host memory'}
            </small>
          </div>
        </section>

        <section className="k8s-section node-details">
          <h3>Deployment details</h3>
          <dl>
            <div><dt>Node</dt><dd>{kubernetes?.node_name || platform?.hostname || 'Unknown'}</dd></div>
            <div><dt>Pool</dt><dd>{kubernetes?.node_pool || 'Local development'}</dd></div>
            <div><dt>Architecture</dt><dd>{kubernetes?.node_architecture || platform?.architecture || 'Unknown'}</dd></div>
            <div><dt>Flex Node</dt><dd>{kubernetes?.flex_node ? 'Verified' : 'Not detected'}</dd></div>
            <div><dt>Pose inference</dt><dd>{result?.execution.processing_device || 'No analysis yet'}</dd></div>
            <div><dt>GPU video</dt><dd>{result?.execution.gpu_video_processing ? result.execution.video_processing_device : 'Inactive'}</dd></div>
          </dl>
        </section>

        <section className="k8s-section activity-section">
          <h3>Current activity</h3>
          <div className="activity-item">
            <i className="status-ready" />
            <div>
              <strong>{result ? 'Analysis result available' : 'Services are ready'}</strong>
              <small>
                {result
                  ? `${result.athlete.pose_frames} pose frames, GPU video ${result.execution.gpu_video_processing ? 'active' : 'inactive'}`
                  : 'Upload a swing to start real 3D inference'}
              </small>
            </div>
          </div>
        </section>
      </div>
    </aside>
  )
}

function KubernetesView({ inventory, platform, result, embedded = false }) {
  const memoryPercent = platform
    ? Math.round((platform.memory_used_mb / platform.memory_total_mb) * 100)
    : 0
  const workloads = [...(inventory?.deployments || []), ...(inventory?.pods || [])]

  return (
    <div className={`kubernetes-page ${embedded ? 'embedded' : ''}`}>
      {!embedded && (
        <div className="kubernetes-page-header">
          <div>
            <h2>AKS Kubernetes Operations</h2>
            <p>Live workload, service, node, and accelerator state for the golf coach deployment.</p>
          </div>
          <span className={`cluster-state ${inventory?.in_cluster ? 'connected' : ''}`}>
            {inventory?.in_cluster ? 'Cluster connected' : 'Local development'}
          </span>
        </div>
      )}

      <div className="cluster-summary">
        <article><span>Node</span><strong>{inventory?.node_name || platform?.hostname || 'Unknown'}</strong><small>{platform?.architecture || 'Reading architecture'}</small></article>
        <article><span>Agent pool</span><strong>{inventory?.node_pool || 'Not in cluster'}</strong><small>{platform?.kubernetes.flex_node ? 'AKS Flex Node verified' : 'Flex Node not detected'}</small></article>
        <article><span>Namespace</span><strong>{inventory?.namespace || 'Not available'}</strong><small>{workloads.length} live workloads</small></article>
        <article className="gpu-summary"><span>NVIDIA GPU</span><strong>{platform?.gpu.available ? platform.gpu.name : 'Not detected'}</strong><small>{platform?.gpu.available ? (platform.gpu.utilization_percent != null ? `${platform.gpu.utilization_percent}% utilization` : 'Utilization not reported') : 'Activates on Jetson Thor'}</small></article>
      </div>

      <div className="kubernetes-dashboard-grid">
        <section className="operations-card workload-card">
          <div className="operations-card-header">
            <div><h3>Workloads</h3><p>Deployments and pods in the application namespace</p></div>
            <span>{workloads.length}</span>
          </div>
          {workloads.length ? (
            <div className="operations-table">
              <div className="operations-table-head"><span>Name</span><span>Kind</span><span>Ready</span><span>Status</span><span>Node</span></div>
              {workloads.map((item) => (
                <div className="operations-table-row" key={`${item.kind}-${item.name}`}>
                  <strong>{item.name}</strong>
                  <span>{item.kind}</span>
                  <span>{item.ready}/{item.desired}</span>
                  <span className={`resource-status ${['Running', 'Available'].includes(item.status) ? 'ready' : ''}`}>{item.status}</span>
                  <span>{item.node_name || 'Controller'}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="cluster-empty-state">
              <strong>No live Kubernetes workload inventory on this host</strong>
              <p>Deploy the application to AKS Flex Node to populate this table with real pods, deployments, readiness, restarts, and scheduling.</p>
            </div>
          )}
        </section>

        <section className="operations-card services-card">
          <div className="operations-card-header">
            <div><h3>Services</h3><p>Live service discovery and network exposure</p></div>
            <span>{inventory?.services.length || 0}</span>
          </div>
          {inventory?.services.length ? inventory.services.map((service) => (
            <article className="service-row" key={service.name}>
              <div><strong>{service.name}</strong><small>{service.cluster_ip || 'No cluster IP'}</small></div>
              <span>{service.service_type}</span>
              <b>{service.ports.join(', ') || 'No ports'}</b>
            </article>
          )) : (
            <div className="declared-resources">
              <p>Declared deployment configuration</p>
              <span><strong>frontend</strong> LoadBalancer, port 80</span>
              <span><strong>backend</strong> ClusterIP, port 8000</span>
            </div>
          )}
        </section>

        <section className="operations-card compute-card">
          <div className="operations-card-header"><div><h3>Node compute</h3><p>Live host and accelerator utilization</p></div><span>Live</span></div>
          <div className="compute-metric">
            <div><span>CPU</span><strong>{platform?.cpu_utilization_percent.toFixed(1) || '0.0'}%</strong></div>
            <div className="metric-track"><i style={{ width: `${platform?.cpu_utilization_percent || 0}%` }} /></div>
            <small>{platform?.cpu_count || 0} logical cores</small>
          </div>
          <div className="compute-metric">
            <div><span>Memory</span><strong>{memoryPercent}%</strong></div>
            <div className="metric-track"><i style={{ width: `${memoryPercent}%` }} /></div>
            <small>{platform ? `${Math.round(platform.memory_used_mb / 1024)} / ${Math.round(platform.memory_total_mb / 1024)} GB` : 'Reading memory'}</small>
          </div>
          <div className="compute-metric">
            <div><span>GPU</span><strong>{platform?.gpu.available && platform.gpu.utilization_percent != null ? `${platform.gpu.utilization_percent}%` : 'N/A'}</strong></div>
            <div className="metric-track"><i className="gpu-fill" style={{ width: `${platform?.gpu.utilization_percent || 0}%` }} /></div>
            <small>{platform?.gpu.available ? gpuMemoryLabel(platform.gpu) : 'NVIDIA GPU not present locally'}</small>
          </div>
        </section>

        <section className="operations-card deployment-card">
          <div className="operations-card-header"><div><h3>Deployment topology</h3><p>Resources declared by the Jetson Thor overlay</p></div></div>
          <div className="topology-flow">
            <span>LoadBalancer</span><i />
            <span>Frontend pod</span><i />
            <span>Backend service</span><i />
            <span>Thor GPU pod</span>
          </div>
          <dl>
            <div><dt>Target node</dt><dd>caldova-gpu-la</dd></div>
            <div><dt>Runtime class</dt><dd>nvidia</dd></div>
            <div><dt>GPU request</dt><dd>nvidia.com/gpu: 1</dd></div>
            <div><dt>GPU video workload</dt><dd>{result?.execution.gpu_video_processing ? 'Active' : 'Waiting for Thor'}</dd></div>
            <div><dt>Service account</dt><dd>golf-coach</dd></div>
          </dl>
        </section>
      </div>
    </div>
  )
}

function UploadPanel({ onAnalyze, status }) {
  const [athlete, setAthlete] = useState(null)
  const [handedness, setHandedness] = useState('right')
  const [dragOver, setDragOver] = useState(false)
  const [videoUrl, setVideoUrl] = useState(null)
  const athleteInput = useRef(null)
  const analyzing = status && !['completed', 'failed'].includes(status.status)

  useEffect(() => {
    if (!athlete) {
      setVideoUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(athlete)
    setVideoUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [athlete])

  useEffect(() => {
    if (analyzing) return undefined
    const handleDragOver = (event) => {
      event.preventDefault()
      setDragOver(true)
    }
    const handleDragLeave = (event) => {
      if (!event.relatedTarget) setDragOver(false)
    }
    const handleDrop = (event) => {
      event.preventDefault()
      setDragOver(false)
      const file = event.dataTransfer.files[0]
      if (file?.type.startsWith('video/')) setAthlete(file)
    }
    window.addEventListener('dragover', handleDragOver)
    window.addEventListener('dragleave', handleDragLeave)
    window.addEventListener('drop', handleDrop)
    return () => {
      window.removeEventListener('dragover', handleDragOver)
      window.removeEventListener('dragleave', handleDragLeave)
      window.removeEventListener('drop', handleDrop)
    }
  }, [analyzing])

  const selectDroppedVideo = (event) => {
    event.preventDefault()
    setDragOver(false)
    const file = event.dataTransfer.files[0]
    if (file?.type.startsWith('video/')) setAthlete(file)
  }

  if (analyzing && videoUrl) {
    return (
      <div className="analysis-progress-page">
        <section className="progress-video-card">
          <div className="progress-video-header">
            <div><strong>Your swing is being analyzed</strong><span>{athlete.name}</span></div>
            <b>Playing source video</b>
          </div>
          <video src={videoUrl} autoPlay muted loop playsInline controls />
        </section>
        <section className="analysis-progress-card">
          <img src="/golf-icon.svg" alt="" />
          <span>3D analysis in progress</span>
          <h2>{status.message}</h2>
          <div className="progress-track"><i style={{ width: `${status.progress}%` }} /></div>
          <div className="analysis-progress-percent">{status.progress}%</div>
          <ul>
            <li className={status.progress >= 10 ? 'complete' : ''}>Decode and sample source video</li>
            <li className={status.progress >= 30 ? 'complete' : ''}>Extract 3D body landmarks</li>
            <li className={status.progress >= 60 ? 'complete' : ''}>Detect eight swing phases</li>
            <li className={status.progress >= 85 ? 'complete' : ''}>Calculate biomechanics and coaching</li>
          </ul>
          <p>The Kubernetes panel shows live infrastructure activity while the source video loops.</p>
        </section>
      </div>
    )
  }

  return (
    <div className="upload-page">
      <div className="upload-intro">
        <h2>Upload Your Golf Swing Video</h2>
        <p>Get a synchronized source video and 3D digital athlete, eight swing phases, measured biomechanics, movement quality, and coaching feedback.</p>
      </div>
      <div className="upload-grid single">
        <button
          className={`upload-area ${athlete ? 'selected' : ''} ${dragOver ? 'dragover' : ''}`}
          onClick={() => athleteInput.current?.click()}
          onDragOver={(event) => { event.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={selectDroppedVideo}
        >
          <img src="/golf-icon.svg" alt="" />
          <strong>{athlete?.name || 'Athlete swing video'}</strong>
          <span>{athlete ? `${(athlete.size / 1048576).toFixed(1)} MB selected` : 'Click to browse or choose a complete swing'}</span>
          <input ref={athleteInput} type="file" accept="video/*" onChange={(event) => setAthlete(event.target.files[0])} />
        </button>
      </div>
      <div className="upload-actions">
        <div className="handedness-control">
          <span>Golfer</span>
          <button className={handedness === 'right' ? 'active' : ''} onClick={() => setHandedness('right')}>Right handed</button>
          <button className={handedness === 'left' ? 'active' : ''} onClick={() => setHandedness('left')}>Left handed</button>
        </div>
        <button
          className="primary-button"
          disabled={!athlete || analyzing}
          onClick={() => onAnalyze({ athlete, handedness })}
        >
          Analyze swing in 3D
        </button>
      </div>
    </div>
  )
}

function ResultsPanel({ result, onReset }) {
  const [selectedPhase, setSelectedPhase] = useState(0)
  const [poseIndex, setPoseIndex] = useState(0)
  const [showGhosts, setShowGhosts] = useState(true)
  const [showAngles, setShowAngles] = useState(true)
  const [videoPlayCount, setVideoPlayCount] = useState(0)
  const videoRef = useRef(null)
  const phase = result.phases[selectedPhase]
  const currentPose = result.pose_sequence[poseIndex] || result.pose_sequence[0]
  const ghostPhases = result.phases.filter((item) =>
    ['address', 'top', 'impact', 'finish'].includes(item.name),
  )

  const synchronize = () => {
    const video = videoRef.current
    if (!video) return
    const time = video.currentTime
    let nextPose = 0
    let poseDistance = Number.POSITIVE_INFINITY
    result.pose_sequence.forEach((item, index) => {
      const distance = Math.abs(item.timestamp_seconds - time)
      if (distance < poseDistance) {
        nextPose = index
        poseDistance = distance
      }
    })
    setPoseIndex(nextPose)
    let nextPhase = 0
    let phaseDistance = Number.POSITIVE_INFINITY
    result.phases.forEach((item, index) => {
      const distance = Math.abs(item.timestamp_seconds - time)
      if (distance < phaseDistance) {
        nextPhase = index
        phaseDistance = distance
      }
    })
    setSelectedPhase(nextPhase)
  }

  const choosePhase = (index) => {
    setSelectedPhase(index)
    if (videoRef.current) {
      videoRef.current.currentTime = result.phases[index].timestamp_seconds
      synchronize()
    }
  }

  const phaseInsights = result.coaching.filter((item) => item.phase === phase.name)
  const visibleInsights = phaseInsights.length ? phaseInsights : result.coaching.slice(0, 2)

  return (
    <div className="results">
      <div className="overall-score">
        <div className={`score-circle ${result.overall_score >= 85 ? 'score-excellent' : result.overall_score >= 70 ? 'score-good' : 'score-fair'}`}>
          {result.overall_score == null ? '3D' : Math.round(result.overall_score)}
        </div>
        <div>
          <h2>{result.overall_score == null ? '3D swing analysis complete' : 'Overall AI Swing Score'}</h2>
          <p>
            {result.athlete.pose_frames} pose frames, {result.execution.processing_seconds} seconds processing,
            {result.tempo_ratio ? ` ${result.tempo_ratio}:1 measured tempo.` : ' tempo not available.'}
          </p>
          {result.execution.gpu_video_processing && (
            <span className="gpu-processing-badge">CUDA video motion analysis active</span>
          )}
        </div>
        <button className="secondary-button" onClick={onReset}>Analyze another</button>
      </div>

      <div className="stage-timeline">
        {result.phases.map((item, index) => (
          <button key={item.name} className={index === selectedPhase ? 'active' : ''} onClick={() => choosePhase(index)}>
            <small>{String(index + 1).padStart(2, '0')}</small>
            <span>{item.display_name}</span>
            {item.score != null && <strong>{Math.round(item.score)}</strong>}
          </button>
        ))}
      </div>

      <div className="comparison-panel">
        <article className="comparison-card">
          <h3>Your Swing <span>{phase.timestamp_seconds.toFixed(2)} seconds</span></h3>
          <video
            ref={videoRef}
            src={`${API_BASE}/api/analyses/${result.analysis_id}/video`}
            controls
            playsInline
            onPlay={() => setVideoPlayCount((count) => count + 1)}
            onTimeUpdate={synchronize}
            onSeeked={synchronize}
          />
        </article>
        <article className="comparison-card model">
          <div className="model-card-title">
            <strong>3D Digital Athlete</strong>
            <div className="model-controls">
              <button className={showGhosts ? 'active' : ''} onClick={() => setShowGhosts(!showGhosts)}>Ghosts</button>
              <button className={showAngles ? 'active' : ''} onClick={() => setShowAngles(!showAngles)}>Angles</button>
              <span>{result.athlete.display_rotation_degrees == null
                ? 'Video angle unavailable. Drag to adjust.'
                : 'Drag to rotate. Play resets view.'}</span>
            </div>
          </div>
          <div className="model-display">
            <div className="model-canvas-wrap">
              <ThreePose
                landmarks={currentPose?.world_landmarks}
                trajectory={result.trajectory}
                ghostLandmarks={showGhosts ? ghostPhases : []}
                handedness={result.handedness || 'right'}
                viewResetKey={videoPlayCount}
                videoRotation={result.athlete.display_rotation_degrees}
              />
              <div className="synchronized-badge">Synchronized with source video</div>
            </div>
            <aside className="model-data-panel">
              {showAngles && (
                <section>
                  <h4>Angles</h4>
                  <dl>
                    <div><dt>Shoulder</dt><dd>{phase.metrics.shoulder_turn.toFixed(1)}°</dd></div>
                    <div><dt>Hips</dt><dd>{phase.metrics.hip_turn.toFixed(1)}°</dd></div>
                    <div><dt>Spine</dt><dd>{phase.metrics.spine_tilt.toFixed(1)}°</dd></div>
                    <div><dt>Lead arm</dt><dd>{phase.metrics.lead_arm_extension.toFixed(1)}°</dd></div>
                    <div><dt>Knee</dt><dd>{phase.metrics.knee_flex.toFixed(1)}°</dd></div>
                  </dl>
                </section>
              )}
              <section>
                <h4>Motion</h4>
                <div className="side-trajectory-legend"><i />Hand path</div>
                {showGhosts && (
                  <div className="side-ghost-legend">
                    <span><i className="address" />Address</span>
                    <span><i className="top" />Top</span>
                    <span><i className="impact" />Impact</span>
                    <span><i className="finish" />Finish</span>
                  </div>
                )}
              </section>
            </aside>
          </div>
        </article>
      </div>

      <div className="feedback-panel">
        <div className="feedback-header">
          <div>
            <span>Stage</span>
            <strong>{phase.display_name}</strong>
          </div>
          <div className="feedback-badges">
            {phase.score != null && <span className="stage-score-badge">Phase score: {Math.round(phase.score)}</span>}
            <span className="confidence-badge">Pose tracking: {Math.round(phase.confidence * 100)}%</span>
          </div>
        </div>
        <div className="feedback-content">
          <div className="measurements">
            <h4>Measured biomechanics</h4>
            <div>
              {Object.entries(phase.metrics).map(([name, value]) => (
                <span key={name}><small>{titleCase(name)}</small><strong>{value.toFixed(1)} degrees</strong></span>
              ))}
            </div>
          </div>
          <div className="coaching-insights">
            <h4>AI coaching insights</h4>
            {visibleInsights.map((insight) => (
              <article key={`${insight.phase}-${insight.title}`} className={insight.severity}>
                <strong>{insight.title}</strong>
                <p>{insight.detail}</p>
                <small>{insight.evidence}</small>
              </article>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [view, setView] = useState('coach')
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [platform, setPlatform] = useState(null)
  const [inventory, setInventory] = useState(null)
  const [error, setError] = useState(null)
  const [k8sOpen, setK8sOpen] = useState(false)

  useEffect(() => {
    const load = () => {
      fetch(`${API_BASE}/api/platform`).then((response) => response.json()).then(setPlatform).catch(() => {})
      fetch(`${API_BASE}/api/kubernetes`).then((response) => response.json()).then(setInventory).catch(() => {})
    }
    load()
    const timer = window.setInterval(load, 5000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!status || ['completed', 'failed'].includes(status.status)) return undefined
    const timer = window.setInterval(async () => {
      const response = await fetch(`${API_BASE}/api/analyses/${status.analysis_id}`)
      const next = await response.json()
      setStatus(next)
      if (next.status === 'completed') {
        const resultResponse = await fetch(`${API_BASE}${next.result_url}`)
        setResult(await resultResponse.json())
        setK8sOpen(true)
      }
      if (next.status === 'failed') setError(next.error)
    }, 1200)
    return () => window.clearInterval(timer)
  }, [status])

  const analyze = async ({ athlete, handedness }) => {
    setError(null)
    setResult(null)
    setK8sOpen(true)
    const form = new FormData()
    form.append('athlete_video', athlete)
    form.append('handedness', handedness)
    const response = await fetch(`${API_BASE}/api/analyses`, { method: 'POST', body: form })
    const data = await response.json()
    if (!response.ok) {
      setError(data.detail || 'The analysis could not be started.')
      return
    }
    setStatus(data)
  }

  const reset = () => {
    setView('coach')
    setStatus(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="header">
        <a
          className="header-home"
          href="/"
          aria-label="Golf Swing AI Coach home"
          onClick={(event) => {
            if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
            event.preventDefault()
            reset()
          }}
        >
          <img src="/golf-icon.svg" alt="" />
          <div className="header-titles">
            <h1>Golf Swing <span>AI Coach</span></h1>
            <p>Local AI on AKS Flex Nodes. Don&apos;t bet your handicap on it.</p>
          </div>
        </a>
        <nav>
          <button className={view === 'coach' ? 'active' : ''} onClick={reset}>Coach</button>
          <button className={view === 'architecture' ? 'active' : ''} onClick={() => { setView('architecture'); setK8sOpen(false) }}>Architecture</button>
        </nav>
      </header>

      <div className="app-body">
        <main>
          {error && <div className="error-message">{error}</div>}
          {view === 'architecture' && (
            <ArchitecturePanel
              kubernetesView={
                <KubernetesView
                  inventory={inventory}
                  platform={platform}
                  result={result}
                  embedded
                />
              }
            />
          )}
          {view === 'coach' && (
            result
              ? <ResultsPanel result={result} onReset={reset} />
              : <UploadPanel onAnalyze={analyze} status={status} />
          )}
        </main>
        {view === 'coach' && (
          <K8sPanel
            isOpen={k8sOpen}
            onToggle={() => setK8sOpen(!k8sOpen)}
            platform={platform}
            result={result}
          />
        )}
      </div>

      <footer>
        <span>Real video, real measurements, live infrastructure telemetry</span>
        <span>AKS Ignite 3D Golf Coach</span>
      </footer>
    </div>
  )
}
