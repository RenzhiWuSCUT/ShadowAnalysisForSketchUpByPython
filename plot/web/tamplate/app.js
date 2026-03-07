(function () {
  const data = window.__THREE_VIEWER_DATA__;
  if (!data) {
    throw new Error("missing window.__THREE_VIEWER_DATA__ (data.js not loaded?)");
  }
  if (!window.THREE) {
    throw new Error("missing THREE (three.min.js not loaded?)");
  }
  if (!THREE.OrbitControls) {
    throw new Error("missing THREE.OrbitControls (OrbitControls.js must be non-ESM version)");
  }

  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(window.devicePixelRatio ? window.devicePixelRatio : 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(55, 2, 0.01, 1e9);
  camera.position.set(10, 10, 10);

  // Z向上
  camera.up.set(0, 0, 1);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // 禁用OrbitControls自带左键旋转（避免它抢行为）
  controls.enableRotate = false;

  controls.zoomSpeed = 0.6;
  controls.panSpeed = 0.08;

  // 旋转速度（越小越慢）
  const ROT_YAW = 0.002;   // 左右：绕Z轴
  const ROT_PITCH = 0.002; // 上下：俯仰
  const MIN_POLAR = 0.08;  // 防止翻到极点(弧度)，越大越“夹紧”

  let isLDown = false;
  let lastX = 0;
  let lastY = 0;

  const AXIS_Z = new THREE.Vector3(0, 0, 1);
  const _offset = new THREE.Vector3();
  const _right = new THREE.Vector3();
  const _tmp = new THREE.Vector3();

  function yawAroundTarget(angleRad) {
    _offset.copy(camera.position).sub(controls.target);
    _offset.applyAxisAngle(AXIS_Z, angleRad);
    camera.position.copy(controls.target).add(_offset);
    camera.lookAt(controls.target);
  }

  function pitchAroundTarget(angleRad) {
    _offset.copy(camera.position).sub(controls.target);

    // 右轴 = up × offset（Z向上）
    _right.crossVectors(AXIS_Z, _offset);
    const len2 = _right.lengthSq();
    if (len2 <= 1e-18) return;
    _right.multiplyScalar(1.0 / Math.sqrt(len2));

    _tmp.copy(_offset).applyAxisAngle(_right, angleRad);

    // clamp：不允许offset接近up方向(极点)
    const n = _tmp.length();
    if (n <= 1e-18) return;
    const cos = THREE.MathUtils.clamp(_tmp.dot(AXIS_Z) / n, -1.0, 1.0);
    const polar = Math.acos(cos); // 与+Z夹角
    if (polar < MIN_POLAR || polar > Math.PI - MIN_POLAR) return;

    camera.position.copy(controls.target).add(_tmp);
    camera.lookAt(controls.target);
  }

  // 通过对角线分区：|dx|>=|dy|认为是“左右区”，否则是“上下区”
  function isHorizontalDrag(dx, dy) {
    const adx = Math.abs(dx);
    const ady = Math.abs(dy);
    return adx >= ady;
  }

  renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

  renderer.domElement.addEventListener("pointerdown", (e) => {
    if (e.button === 0) {
      isLDown = true;
      lastX = e.clientX;
      lastY = e.clientY;
      renderer.domElement.setPointerCapture(e.pointerId);
      e.preventDefault();
      e.stopPropagation();
    }
  });

  renderer.domElement.addEventListener("pointerup", (e) => {
    if (e.button === 0) {
      isLDown = false;
      renderer.domElement.releasePointerCapture(e.pointerId);
      e.preventDefault();
      e.stopPropagation();
    }
  });

  renderer.domElement.addEventListener("pointermove", (e) => {
    if (!isLDown) return;

    const x = e.clientX;
    const y = e.clientY;
    const dx = x - lastX;
    const dy = y - lastY;

    lastX = x;
    lastY = y;

    if (dx === 0 && dy === 0) return;

    if (isHorizontalDrag(dx, dy)) {
      yawAroundTarget(-dx * ROT_YAW);
    } else {
      pitchAroundTarget(-dy * ROT_PITCH);
    }

    controls.update();

    e.preventDefault();
    e.stopPropagation();
  });

  const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 0.9);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(10, 30, 20);
  scene.add(dir);

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const need = (canvas.width !== w) || (canvas.height !== h);
    if (need) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }

  function makeLine(points, color) {
    const geom = new THREE.BufferGeometry();
    const arr = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      arr[i * 3 + 0] = points[i][0];
      arr[i * 3 + 1] = points[i][1];
      arr[i * 3 + 2] = points[i][2];
    }
    geom.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    const mat = new THREE.LineBasicMaterial({ color: color });
    return new THREE.Line(geom, mat);
  }

  function closeLoopPts(loopPts) {
    if (!loopPts || loopPts.length < 2) return loopPts;
    const a = loopPts[0];
    const b = loopPts[loopPts.length - 1];
    if (a[0] === b[0] && a[1] === b[1] && a[2] === b[2]) return loopPts;
    const out = loopPts.slice();
    out.push([a[0], a[1], a[2]]);
    return out;
  }

  function stripClosing(loopPts) {
    if (!loopPts || loopPts.length < 3) return loopPts;
    const a = loopPts[0];
    const b = loopPts[loopPts.length - 1];
    if (a[0] === b[0] && a[1] === b[1] && a[2] === b[2]) {
      return loopPts.slice(0, loopPts.length - 1);
    }
    return loopPts;
  }

  // 在quad4定义的面内基底上，把点P投影成(a,b)，满足：P = p0 + a*u + b*v
  // 这里u=quad4[1]-quad4[0], v=quad4[3]-quad4[0]，且UV=(a,b)，对应纹理[0..1]
  function makeQuadProjector(quad4) {
    const p0 = new THREE.Vector3(quad4[0][0], quad4[0][1], quad4[0][2]);
    const p1 = new THREE.Vector3(quad4[1][0], quad4[1][1], quad4[1][2]);
    const p3 = new THREE.Vector3(quad4[3][0], quad4[3][1], quad4[3][2]);

    const u = p1.clone().sub(p0);
    const v = p3.clone().sub(p0);

    const uu = u.dot(u);
    const uv = u.dot(v);
    const vv = v.dot(v);

    const det = uu * vv - uv * uv;
    const eps = 1e-20;

    function project(P) {
      const d = P.clone().sub(p0);
      const du = d.dot(u);
      const dv = d.dot(v);

      if (Math.abs(det) <= eps) {
        // 退化：退回到简单投影
        const a = (uu > eps) ? (du / uu) : 0.0;
        const b = (vv > eps) ? (dv / vv) : 0.0;
        return new THREE.Vector2(a, b);
      }

      // 解Gram矩阵： [uu uv; uv vv] [a b]^T = [du dv]^T
      const invDet = 1.0 / det;
      const a = (vv * du - uv * dv) * invDet;
      const b = (-uv * du + uu * dv) * invDet;
      return new THREE.Vector2(a, b);
    }

    return { project };
  }

  // =============== 关键新增：面自身平面投影用于三角剖分 ===============
  function newellNormal(pts3) {
    // pts3: Array<[x,y,z]>，不闭合
    const n = new THREE.Vector3(0, 0, 0);
    const m = pts3.length;
    for (let i = 0; i < m; i++) {
      const p0 = pts3[i];
      const p1 = pts3[(i + 1) % m];
      n.x += (p0[1] - p1[1]) * (p0[2] + p1[2]);
      n.y += (p0[2] - p1[2]) * (p0[0] + p1[0]);
      n.z += (p0[0] - p1[0]) * (p0[1] + p1[1]);
    }
    const len = n.length();
    if (len <= 1e-18) return new THREE.Vector3(0, 0, 0);
    return n.multiplyScalar(1.0 / len);
  }

  function makeFacePlaneProjector(outer3, quad4) {
    // 返回：project(P)->Vector2（用于剖分），以及 {uAxis,vAxis,n,p0}
    let n = newellNormal(outer3);
    if (n.lengthSq() <= 1e-18 && quad4 && quad4.length >= 4) {
      // Newell退化：用quad4兜底
      const q0 = new THREE.Vector3(quad4[0][0], quad4[0][1], quad4[0][2]);
      const q1 = new THREE.Vector3(quad4[1][0], quad4[1][1], quad4[1][2]);
      const q3 = new THREE.Vector3(quad4[3][0], quad4[3][1], quad4[3][2]);
      const u = q1.clone().sub(q0);
      const v = q3.clone().sub(q0);
      n = u.clone().cross(v);
      const len = n.length();
      if (len > 1e-18) n.multiplyScalar(1.0 / len);
      else n.set(0, 0, 1);
    }

    const ref = (Math.abs(n.z) < 0.9) ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    const uAxis = ref.clone().cross(n);
    const uLen = uAxis.length();
    if (uLen <= 1e-18) {
      // 极端兜底
      uAxis.set(1, 0, 0);
    } else {
      uAxis.multiplyScalar(1.0 / uLen);
    }
    const vAxis = n.clone().cross(uAxis); // 已单位

    const p0 = new THREE.Vector3(outer3[0][0], outer3[0][1], outer3[0][2]);

    function project(P) {
      const d = P.clone().sub(p0);
      const x = d.dot(uAxis);
      const y = d.dot(vAxis);
      return new THREE.Vector2(x, y);
    }

    return { project, uAxis, vAxis, n, p0 };
  }

  function toVec2ArrByProject(loopPts3, projectorFn) {
    const out = [];
    for (let i = 0; i < loopPts3.length; i++) {
      const p = loopPts3[i];
      const P = new THREE.Vector3(p[0], p[1], p[2]);
      out.push(projectorFn(P));
    }
    return out;
  }

  // 真实面：按outer+holes三角剖分，贴图仍按quad坐标系对齐
  function makeFaceMesh(f, opacity) {
    const outerSrc = stripClosing(f.outer || []);
    if (!outerSrc || outerSrc.length < 3) return null;

    const holesSrc = (f.holes || []).map(stripClosing).filter(h => h && h.length >= 3);

    // 1) 剖分：用面自身平面坐标系（稳定，不受竖直/退化quad影响）
    const faceProj = makeFacePlaneProjector(outerSrc, f.quad4);
    let contour2_geom = toVec2ArrByProject(outerSrc, faceProj.project);
    let holes2_geom = holesSrc.map(h3 => toVec2ArrByProject(h3, faceProj.project));

    // 2) UV：仍用quad4投影（保证贴图与bound对齐）
    const uvProj = makeQuadProjector(f.quad4);
    let contour2_uv = toVec2ArrByProject(outerSrc, uvProj.project);
    let holes2_uv = holesSrc.map(h3 => toVec2ArrByProject(h3, uvProj.project));

    // 需要能同步反转：外轮廓顺时针，洞逆时针（以“剖分几何2D”为准）
    let outer3 = outerSrc.slice(); // 不污染输入
    let holes3 = holesSrc.map(h => h.slice());

    if (!THREE.ShapeUtils.isClockWise(contour2_geom)) {
      contour2_geom = contour2_geom.slice().reverse();
      contour2_uv = contour2_uv.slice().reverse();
      outer3 = outer3.slice().reverse();
    }
    for (let i = 0; i < holes2_geom.length; i++) {
      if (THREE.ShapeUtils.isClockWise(holes2_geom[i])) {
        holes2_geom[i] = holes2_geom[i].slice().reverse();
        holes2_uv[i] = holes2_uv[i].slice().reverse();
        holes3[i] = holes3[i].slice().reverse();
      }
    }

    const triangles = THREE.ShapeUtils.triangulateShape(contour2_geom, holes2_geom);
    if (!triangles || triangles.length === 0) return null;

    // 顶点顺序必须与triangulateShape输出索引一致：vertices = contour + holes(flat)
    const verts3 = [];
    const verts2_uv = []; // 用于写入UV

    for (let i = 0; i < outer3.length; i++) {
      const p = outer3[i];
      verts3.push(new THREE.Vector3(p[0], p[1], p[2]));
      verts2_uv.push(contour2_uv[i]);
    }
    for (let h = 0; h < holes3.length; h++) {
      const h3 = holes3[h];
      const hUv = holes2_uv[h];
      for (let i = 0; i < h3.length; i++) {
        const p = h3[i];
        verts3.push(new THREE.Vector3(p[0], p[1], p[2]));
        verts2_uv.push(hUv[i]);
      }
    }

    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(verts3.length * 3);
    const uv = new Float32Array(verts3.length * 2);

    for (let i = 0; i < verts3.length; i++) {
      const P = verts3[i];
      pos[i * 3 + 0] = P.x;
      pos[i * 3 + 1] = P.y;
      pos[i * 3 + 2] = P.z;

      const T = verts2_uv[i];

      let u = 0.0;
      let v = 0.0;

      if (T && Number.isFinite(T.x) && Number.isFinite(T.y)) {
        u = T.x;
        v = T.y;

        // 安全夹取，避免极端值
        if (u < -8) u = -8;
        if (u > 8) u = 8;
        if (v < -8) v = -8;
        if (v > 8) v = 8;
      } else {
        u = 0.0;
        v = 0.0;
      }

      uv[i * 2 + 0] = u;
      uv[i * 2 + 1] = v;
    }

    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("uv", new THREE.BufferAttribute(uv, 2));

    const use32 = verts3.length > 65535;
    const index = use32 ? new Uint32Array(triangles.length * 3) : new Uint16Array(triangles.length * 3);
    for (let i = 0; i < triangles.length; i++) {
      index[i * 3 + 0] = triangles[i][0];
      index[i * 3 + 1] = triangles[i][1];
      index[i * 3 + 2] = triangles[i][2];
    }
    geom.setIndex(new THREE.BufferAttribute(index, 1));
    geom.computeVertexNormals();

    const tex = new THREE.TextureLoader().load(f.tex);
    tex.flipY = true;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;

    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      transparent: opacity < 1.0,
      opacity: opacity,
      side: THREE.DoubleSide,
      depthWrite: opacity >= 1.0,
    });

    return new THREE.Mesh(geom, mat);
  }

  function fitCameraToBox(box) {
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * Math.PI / 180;
    let dist = maxDim / (2 * Math.tan(fov / 2));
    dist *= 1.6;

    const vdir = new THREE.Vector3(1, 1, 1).normalize();
    camera.position.copy(center.clone().add(vdir.multiplyScalar(dist)));
    camera.near = Math.max(0.001, dist / 1000);
    camera.far = dist * 1000;
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.update();
  }

  const root = new THREE.Group();
  scene.add(root);

  const bbox = new THREE.Box3();

  const faces = data.faces || [];
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];

    const mesh = makeFaceMesh(f, data.quad_opacity || 1.0);
    if (mesh) {
      root.add(mesh);
      bbox.expandByObject(mesh);
    }

    // 线框显示（可选）
    if (data.show_outline) {
      if (f.outer && f.outer.length >= 2) {
        const pts = closeLoopPts(f.outer);
        const ln = makeLine(pts, data.outline_color || "#ffffff");
        root.add(ln);
      }
      const holes = f.holes || [];
      for (let h = 0; h < holes.length; h++) {
        const hp = holes[h];
        if (hp && hp.length >= 2) {
          const pts = closeLoopPts(hp);
          const ln = makeLine(pts, data.hole_color || "#ffcc00");
          root.add(ln);
        }
      }
    }
  }

  if (!bbox.isEmpty()) {
    fitCameraToBox(bbox);
  }

  if (window.buildColorBar) {
    window.buildColorBar(data);
  }

  function animate() {
    resize();
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();

  window.addEventListener("resize", resize);
})();