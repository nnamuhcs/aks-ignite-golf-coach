import { useEffect, useRef } from 'react'
import * as THREE from 'three'

const LIMBS = [
  { start: 11, end: 13, radius: 0.048, group: 'upper' },
  { start: 13, end: 15, radius: 0.04, group: 'upper' },
  { start: 12, end: 14, radius: 0.048, group: 'upper' },
  { start: 14, end: 16, radius: 0.04, group: 'upper' },
  { start: 23, end: 25, radius: 0.064, group: 'lower' },
  { start: 25, end: 27, radius: 0.052, group: 'lower' },
  { start: 24, end: 26, radius: 0.064, group: 'lower' },
  { start: 26, end: 28, radius: 0.052, group: 'lower' },
  { start: 27, end: 31, radius: 0.036, group: 'lower' },
  { start: 28, end: 32, radius: 0.036, group: 'lower' },
]

const JOINTS = [13, 14, 15, 16, 25, 26, 27, 28, 31, 32]
const TORSO_INDICES = [11, 12, 24, 23]
const GHOST_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [27, 31], [24, 26], [26, 28], [28, 32],
]

function updatePanelGeometry(geometry, points, indices) {
  const positions = geometry.attributes.position.array
  indices.forEach((pointIndex, index) => {
    positions[index * 3] = points[pointIndex].x
    positions[index * 3 + 1] = points[pointIndex].y
    positions[index * 3 + 2] = points[pointIndex].z
  })
  geometry.attributes.position.needsUpdate = true
  geometry.computeVertexNormals()
}

function positionBone(bone, start, end) {
  const direction = new THREE.Vector3().subVectors(end, start)
  const length = direction.length()
  bone.position.copy(start).add(end).multiplyScalar(0.5)
  bone.scale.set(1, length, 1)
  bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize())
}

function createPanelGeometry() {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3))
  geometry.setIndex([0, 1, 2, 0, 2, 3])
  return geometry
}

function createLabelSprite() {
  const canvas = document.createElement('canvas')
  canvas.width = 260
  canvas.height = 56
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      opacity: 0.86,
    }),
  )
  sprite.scale.set(0.38, 0.082, 1)
  sprite.userData = { canvas, texture }
  return sprite
}

function updateLabelSprite(sprite, label) {
  const { canvas, texture } = sprite.userData
  const context = canvas.getContext('2d')
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.fillStyle = 'rgba(4, 18, 29, 0.88)'
  context.beginPath()
  context.roundRect(2, 2, canvas.width - 4, canvas.height - 4, 12)
  context.fill()
  context.strokeStyle = 'rgba(91, 232, 196, 0.9)'
  context.lineWidth = 2
  context.stroke()
  context.fillStyle = '#dffff7'
  context.font = '600 20px Inter, Arial, sans-serif'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(label, canvas.width / 2, canvas.height / 2)
  texture.needsUpdate = true
}

function createGhost(points, color) {
  const group = new THREE.Group()
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.17,
    depthWrite: false,
  })
  GHOST_CONNECTIONS.forEach(([start, end]) => {
    const bone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.022, 0.022, 1, 8),
      material,
    )
    positionBone(bone, points[start], points[end])
    group.add(bone)
  })
  const jointGeometry = new THREE.SphereGeometry(0.034, 10, 8)
  JOINTS.forEach((pointIndex) => {
    const joint = new THREE.Mesh(jointGeometry, material)
    joint.position.copy(points[pointIndex])
    group.add(joint)
  })
  const torsoGeometry = createPanelGeometry()
  updatePanelGeometry(torsoGeometry, points, TORSO_INDICES)
  group.add(new THREE.Mesh(torsoGeometry, material))
  const headCenter = new THREE.Vector3().addVectors(points[7], points[8]).multiplyScalar(0.5)
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.095, 14, 10), material)
  head.position.copy(headCenter)
  group.add(head)
  return group
}

export default function ThreePose({
  landmarks,
  trajectory,
  ghostLandmarks = [],
  angleOverlays = null,
  handedness = 'right',
}) {
  const mountRef = useRef(null)
  const athleteRef = useRef(null)
  const orientationRef = useRef(null)

  useEffect(() => {
    if (!mountRef.current) return undefined

    const mount = mountRef.current
    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x07111f, 0.14)

    const camera = new THREE.PerspectiveCamera(34, mount.clientWidth / mount.clientHeight, 0.1, 100)
    camera.position.set(0, 0.15, 4.4)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    mount.appendChild(renderer.domElement)

    const athlete = new THREE.Group()
    athlete.position.y = 0
    scene.add(athlete)

    const upperMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x36e2bd,
      emissive: 0x063c35,
      metalness: 0.28,
      roughness: 0.2,
      clearcoat: 0.75,
      clearcoatRoughness: 0.18,
    })
    const lowerMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x3a82ff,
      emissive: 0x071b4d,
      metalness: 0.35,
      roughness: 0.24,
      clearcoat: 0.7,
    })
    const jointMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xe9fffb,
      emissive: 0x1a6259,
      metalness: 0.45,
      roughness: 0.12,
      clearcoat: 1,
    })
    const coreMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x21b998,
      emissive: 0x063d35,
      metalness: 0.2,
      roughness: 0.25,
      clearcoat: 0.8,
      side: THREE.DoubleSide,
    })
    const headMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x9fffea,
      emissive: 0x0b5549,
      transparent: true,
      opacity: 0.92,
      metalness: 0.15,
      roughness: 0.1,
      transmission: 0.12,
      clearcoat: 1,
    })

    const torsoGeometry = createPanelGeometry()
    const torso = new THREE.Mesh(torsoGeometry, coreMaterial)
    torso.castShadow = true
    athlete.add(torso)

    const pelvisGeometry = createPanelGeometry()
    const pelvis = new THREE.Mesh(pelvisGeometry, lowerMaterial)
    pelvis.castShadow = true
    athlete.add(pelvis)

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.115, 32, 24), headMaterial)
    head.scale.set(0.82, 1.12, 0.88)
    head.castShadow = true
    athlete.add(head)

    const neck = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.065, 1, 18),
      upperMaterial,
    )
    neck.castShadow = true
    athlete.add(neck)

    const limbs = LIMBS.map((definition) => {
      const material = definition.group === 'upper' ? upperMaterial : lowerMaterial
      const limb = new THREE.Mesh(
        new THREE.CylinderGeometry(
          definition.radius * 0.78,
          definition.radius,
          1,
          18,
        ),
        material,
      )
      limb.castShadow = true
      athlete.add(limb)
      return limb
    })

    const jointGeometry = new THREE.SphereGeometry(0.055, 20, 16)
    const joints = JOINTS.map((index) => {
      const joint = new THREE.Mesh(jointGeometry, jointMaterial)
      const extremity = [15, 16, 31, 32].includes(index)
      joint.scale.setScalar(extremity ? 0.82 : 1)
      joint.castShadow = true
      athlete.add(joint)
      return joint
    })
    const angleSprites = Array.from({ length: 5 }, () => {
      const sprite = createLabelSprite()
      sprite.visible = false
      athlete.add(sprite)
      return sprite
    })

    athleteRef.current = {
      athlete,
      torsoGeometry,
      pelvisGeometry,
      head,
      neck,
      limbs,
      joints,
      angleSprites,
      ghostGroups: [],
      trajectoryPath: null,
    }
    orientationRef.current = null

    const floorGlow = new THREE.Mesh(
      new THREE.CircleGeometry(1.45, 64),
      new THREE.MeshBasicMaterial({
        color: 0x0b6f65,
        transparent: true,
        opacity: 0.12,
        depthWrite: false,
      }),
    )
    floorGlow.rotation.x = -Math.PI / 2
    floorGlow.position.y = -0.96
    scene.add(floorGlow)

    const floor = new THREE.GridHelper(4, 16, 0x2a927f, 0x123148)
    floor.position.y = -0.95
    floor.material.transparent = true
    floor.material.opacity = 0.42
    scene.add(floor)

    scene.add(new THREE.HemisphereLight(0xd9fff7, 0x04101d, 2.4))
    const key = new THREE.DirectionalLight(0x81ffe3, 5.5)
    key.position.set(2.5, -3, 4)
    key.castShadow = true
    scene.add(key)
    const rim = new THREE.PointLight(0x3c7dff, 18, 8)
    rim.position.set(-2.5, 0, -1.5)
    scene.add(rim)
    const warm = new THREE.PointLight(0xffbd63, 10, 7)
    warm.position.set(2, 1, 2)
    scene.add(warm)

    let frameId
    let dragging = false
    let previousX = 0
    const onPointerDown = (event) => {
      dragging = true
      previousX = event.clientX
      renderer.domElement.setPointerCapture(event.pointerId)
    }
    const onPointerMove = (event) => {
      if (!dragging) return
      athlete.rotation.y += (event.clientX - previousX) * 0.01
      previousX = event.clientX
    }
    const onPointerUp = () => {
      dragging = false
    }
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointermove', onPointerMove)
    renderer.domElement.addEventListener('pointerup', onPointerUp)

    const animate = () => {
      frameId = requestAnimationFrame(animate)
      renderer.render(scene, camera)
    }
    animate()

    const observer = new ResizeObserver(() => {
      if (!mount.clientWidth || !mount.clientHeight) return
      camera.aspect = mount.clientWidth / mount.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mount.clientWidth, mount.clientHeight)
    })
    observer.observe(mount)

    return () => {
      observer.disconnect()
      cancelAnimationFrame(frameId)
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      const geometries = new Set()
      const materials = new Set()
      scene.traverse((object) => {
        if (object.geometry) geometries.add(object.geometry)
        if (object.material) {
          const objectMaterials = Array.isArray(object.material) ? object.material : [object.material]
          objectMaterials.forEach((material) => materials.add(material))
        }
      })
      geometries.forEach((geometry) => geometry.dispose())
      materials.forEach((material) => material.dispose())
      renderer.dispose()
      athleteRef.current = null
      orientationRef.current = null
      mount.removeChild(renderer.domElement)
    }
  }, [trajectory])

  useEffect(() => {
    if (!landmarks?.length || !athleteRef.current) return
    const rawPoints = landmarks.map((point) => new THREE.Vector3(point.x, point.y, point.z))
    if (!orientationRef.current) {
      const hipCenter = new THREE.Vector3().addVectors(rawPoints[23], rawPoints[24]).multiplyScalar(0.5)
      const ankleCenter = new THREE.Vector3().addVectors(rawPoints[27], rawPoints[28]).multiplyScalar(0.5)
      const down = new THREE.Vector3().subVectors(ankleCenter, hipCenter).normalize()
      orientationRef.current = new THREE.Quaternion().setFromUnitVectors(
        down,
        new THREE.Vector3(0, -1, 0),
      )
    }
    const orientation = orientationRef.current
    const points = rawPoints.map((point) => point.clone().applyQuaternion(orientation))
    const {
      athlete,
      torsoGeometry,
      pelvisGeometry,
      head,
      neck,
      limbs,
      joints,
      angleSprites,
    } = athleteRef.current

    if (!athleteRef.current.trajectoryPath && trajectory?.length > 3) {
      const pathPoints = trajectory.map(
        (point) => new THREE.Vector3(point.x, point.y, point.z).applyQuaternion(orientation),
      )
      const curve = new THREE.CatmullRomCurve3(pathPoints)
      const path = new THREE.Mesh(
        new THREE.TubeGeometry(curve, Math.min(pathPoints.length * 2, 320), 0.009, 8, false),
        new THREE.MeshBasicMaterial({
          color: 0xffc653,
          transparent: true,
          opacity: 0.72,
        }),
      )
      athlete.add(path)
      athleteRef.current.trajectoryPath = path
    }

    updatePanelGeometry(torsoGeometry, points, TORSO_INDICES)

    const leftUpperLeg = new THREE.Vector3().lerpVectors(points[23], points[25], 0.24)
    const rightUpperLeg = new THREE.Vector3().lerpVectors(points[24], points[26], 0.24)
    const pelvisPoints = [...points]
    pelvisPoints.push(leftUpperLeg, rightUpperLeg)
    updatePanelGeometry(pelvisGeometry, pelvisPoints, [23, 24, 34, 33])

    const earCenter = new THREE.Vector3().addVectors(points[7], points[8]).multiplyScalar(0.5)
    const shoulderCenter = new THREE.Vector3().addVectors(points[11], points[12]).multiplyScalar(0.5)
    head.position.copy(earCenter)
    const neckEnd = new THREE.Vector3().lerpVectors(shoulderCenter, earCenter, 0.62)
    positionBone(neck, shoulderCenter, neckEnd)

    LIMBS.forEach((definition, index) => {
      positionBone(limbs[index], points[definition.start], points[definition.end])
    })
    JOINTS.forEach((pointIndex, index) => {
      joints[index].position.copy(points[pointIndex])
    })

    if (!athleteRef.current.ghostGroups.length && ghostLandmarks.length) {
      const colors = [0x63e6be, 0x60a5fa, 0xf6c453, 0xa78bfa]
      athleteRef.current.ghostGroups = ghostLandmarks.map((ghost, index) => {
        const ghostPoints = ghost.world_landmarks.map(
          (point) => new THREE.Vector3(point.x, point.y, point.z).applyQuaternion(orientation),
        )
        const group = createGhost(ghostPoints, colors[index % colors.length])
        athlete.add(group)
        return group
      })
    }
    athleteRef.current.ghostGroups.forEach((group) => {
      group.visible = ghostLandmarks.length > 0
    })

    angleSprites.forEach((sprite) => {
      sprite.visible = Boolean(angleOverlays)
    })
    if (angleOverlays) {
      const hipCenter = new THREE.Vector3().addVectors(points[23], points[24]).multiplyScalar(0.5)
      const spineCenter = new THREE.Vector3().addVectors(shoulderCenter, hipCenter).multiplyScalar(0.5)
      const leadElbowIndex = handedness === 'left' ? 14 : 13
      const kneeCenter = new THREE.Vector3().addVectors(points[25], points[26]).multiplyScalar(0.5)
      const labels = [
        [`Shoulder ${angleOverlays.shoulder_turn.toFixed(1)}°`, shoulderCenter.clone().add(new THREE.Vector3(-0.27, 0.1, 0))],
        [`Hips ${angleOverlays.hip_turn.toFixed(1)}°`, hipCenter.clone().add(new THREE.Vector3(0.27, 0, 0))],
        [`Spine ${angleOverlays.spine_tilt.toFixed(1)}°`, spineCenter.clone().add(new THREE.Vector3(0.27, 0.1, 0))],
        [`Lead ${angleOverlays.lead_arm_extension.toFixed(1)}°`, points[leadElbowIndex].clone().add(new THREE.Vector3(-0.26, 0, 0))],
        [`Knee ${angleOverlays.knee_flex.toFixed(1)}°`, kneeCenter.clone().add(new THREE.Vector3(0.26, -0.04, 0))],
      ]
      labels.forEach(([label, position], index) => {
        updateLabelSprite(angleSprites[index], label)
        angleSprites[index].position.copy(position)
      })
    }
  }, [landmarks, trajectory, ghostLandmarks, angleOverlays, handedness])

  return <div className="three-pose" ref={mountRef} />
}
