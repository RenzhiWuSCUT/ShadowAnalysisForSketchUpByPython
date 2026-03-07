# readSU.py
from __future__ import annotations
import os
import ctypes
from ctypes import c_int, c_size_t, byref
from obj.Geo import SketchUpGeo, Point3, Edge, Loop, Face, Material
import obj._SU_API as SU

_UNIT_TO_M = {
    "Inches": 0.0254,
    "Feet": 0.3048,
    "Millimeters": 0.001,
    "Centimeters": 0.01,
    "Meters": 1.0,
}


# --------- Pretty print ---------
def _line(ch: str = "-", n: int = 84) -> str:
    return ch * n


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _box(title: str, rows: list[tuple[str, str]], width: int = 84) -> str:
    k_w = 18
    v_w = width - 6 - k_w
    out = []
    out.append("+" + _line("-", width - 2) + "+")
    t = f" {title} "
    pad_left = max(0, (width - 2 - len(t)) // 2)
    pad_right = max(0, (width - 2 - len(t)) - pad_left)
    out.append("|" + " " * pad_left + t[: width - 2] + " " * pad_right + "|")
    out.append("+" + _line("-", width - 2) + "+")
    for k, v in rows:
        k2 = (k[:k_w]).ljust(k_w)
        vs = str(v).splitlines() if str(v) else [""]
        first = True
        for vi in vs:
            vi2 = (vi[:v_w]).ljust(v_w)
            if first:
                out.append(f"| {k2} | {vi2} |")
                first = False
            else:
                out.append(f"| {'':{k_w}} | {vi2} |")
    out.append("+" + _line("-", width - 2) + "+")
    return "\n".join(out)


def format_mesh_report(mesh: SketchUpGeo, width: int = 84, show_materials: int = 10) -> str:
    head = _box(
        "SketchUp Mesh读取结果",
        [
            ("文件", mesh.file_path),
            ("单位", mesh.units if mesh.units else "(未知)"),
            ("SUInitialize", str(mesh.su_initialize_result)),
            ("Load SUResult", str(mesh.load_result)),
            ("LoadStatus", str(mesh.load_status)),
            ("Model ptr", str(mesh.model_ptr)),
            ("模型名", mesh.model_name if mesh.model_name else "(空)"),
        ],
        width=width,
    )
    stats = _box(
        "几何统计",
        [
            ("points", _fmt_int(len(mesh.points))),
            ("edges", _fmt_int(len(mesh.edges))),
            ("loops", _fmt_int(len(mesh.loops))),
            ("faces", _fmt_int(len(mesh.faces))),
            ("materials", _fmt_int(len(mesh.materials))),
            ("colors", _fmt_int(len(mesh.colors))),
            ("textures", _fmt_int(len(mesh.textures))),
        ],
        width=width,
    )
    mats = ""
    if len(mesh.materials) > 0:
        n = min(show_materials, len(mesh.materials))
        lines = []
        for i in range(n):
            m = mesh.materials[i]
            nm = m.name if m.name else "(空名)"
            ref = "none"
            if m.has_color():
                ref = f"color[{m.color_index}]"
            elif m.has_texture():
                ref = f"tex[{m.texture_index}]"
            lines.append(f"{i:>3}: {nm} -> {ref}")
        if len(mesh.materials) > n:
            lines.append(f"...(还有{len(mesh.materials) - n}个)")
        mats = _box("材质样例", [("materials", "\n".join(lines))], width=width)

    return "\n\n".join([head, stats, mats]).strip()


# --------- Reader ---------
class SketchUpReader:
    def __init__(self, dll_path: str):
        self.dll_path = os.path.abspath(dll_path)
        if not os.path.isfile(self.dll_path):
            raise FileNotFoundError(self.dll_path)

        dll_dir = os.path.dirname(self.dll_path)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

        self._old_cwd = os.getcwd()
        os.chdir(dll_dir)

        self.su = ctypes.CDLL(self.dll_path)
        SU.bind_core_api(self.su)

    def close(self) -> None:
        os.chdir(self._old_cwd)

    def _get_model_name(self, model: SU.SUModelRef) -> str:
        if not (SU.has(self.su, "SUModelGetName") and SU.has(self.su, "SUStringCreate")):
            return ""
        s = SU.SUStringRef()
        SU.su_check(self.su.SUStringCreate(byref(s)), "SUStringCreate")
        SU.su_check(self.su.SUModelGetName(model, byref(s)), "SUModelGetName")
        name = SU.su_string_to_py(self.su, s)
        SU.su_check(self.su.SUStringRelease(byref(s)), "SUStringRelease")
        return name

    def _get_units(self, model: SU.SUModelRef) -> str:
        if not SU.has(self.su, "SUModelGetUnits"):
            return ""
        u = c_int(0)
        res = int(self.su.SUModelGetUnits(model, byref(u)))
        if res != SU.SU_ERROR_NONE:
            return ""
        mp = {
            0: "Inches",
            1: "Feet",
            2: "Millimeters",
            3: "Centimeters",
            4: "Meters",
        }
        return mp.get(int(u.value), f"UnitsEnum({int(u.value)})")

    def _material_to_index(self, geo: SketchUpGeo, mat: SU.SUMaterialRef) -> int:
        if not mat.ptr:
            return -1
        mptr = int(mat.ptr)
        for i, m in enumerate(geo.materials):
            if m.ptr == mptr and mptr != 0:
                return i

        name = ""
        if SU.has(self.su, "SUMaterialGetName") and SU.has(self.su, "SUStringCreate"):
            s = SU.SUStringRef()
            SU.su_check(self.su.SUStringCreate(byref(s)), "SUStringCreate")
            SU.su_check(self.su.SUMaterialGetName(mat, byref(s)), "SUMaterialGetName")
            name = SU.su_string_to_py(self.su, s)
            SU.su_check(self.su.SUStringRelease(byref(s)), "SUStringRelease")

        geo.materials.append(Material(name=name, ptr=mptr))
        return len(geo.materials) - 1

    def _point_index(self, geo: SketchUpGeo, p: SU.SUPoint3D, point_map: dict[tuple[int, int, int], int]) -> int:
        key = (
            int(round(float(p.x) * 1_000_000)),
            int(round(float(p.y) * 1_000_000)),
            int(round(float(p.z) * 1_000_000)),
        )
        idx = point_map.get(key)
        if idx is not None:
            return idx
        geo.points.append(Point3(float(p.x), float(p.y), float(p.z)))
        idx2 = len(geo.points) - 1
        point_map[key] = idx2
        return idx2

    def _edge_index(self, geo: SketchUpGeo, a: int, b: int, edge_map: dict[tuple[int, int], int]) -> int:
        k = (a, b) if a < b else (b, a)
        idx = edge_map.get(k)
        if idx is not None:
            return idx
        geo.edges.append(Edge(a=a, b=b))
        idx2 = len(geo.edges) - 1
        edge_map[k] = idx2
        return idx2

    def _extract_loop(
            self,
            geo: SketchUpGeo,
            loop: SU.SULoopRef,
            is_outer: bool,
            point_map: dict[tuple[int, int, int], int],
            edge_map: dict[tuple[int, int], int],
    ) -> int:
        vn = c_size_t(0)
        SU.su_check(self.su.SULoopGetNumVertices(loop, byref(vn)), "SULoopGetNumVertices")
        cnt = int(vn.value)
        if cnt <= 0:
            geo.loops.append(Loop(is_outer=is_outer, edges=[], verts=[]))
            return len(geo.loops) - 1

        verts_ref = (SU.SUVertexRef * cnt)()
        got = c_size_t(0)
        SU.su_check(self.su.SULoopGetVertices(loop, cnt, verts_ref, byref(got)), "SULoopGetVertices")
        n = int(got.value)

        ring: list[int] = []
        for i in range(n):
            pos = SU.SUPoint3D()
            SU.su_check(self.su.SUVertexGetPosition(verts_ref[i], byref(pos)), "SUVertexGetPosition")
            ring.append(self._point_index(geo, pos, point_map))

        eidx: list[int] = []
        if len(ring) >= 2:
            for i in range(len(ring)):
                a = ring[i]
                b = ring[(i + 1) % len(ring)]
                eidx.append(self._edge_index(geo, a, b, edge_map))

        geo.loops.append(Loop(is_outer=is_outer, edges=eidx, verts=ring))
        return len(geo.loops) - 1

    def _face_normal_from_outer_loop(self, geo: SketchUpGeo, outer_loop_idx: int) -> Point3:
        lp = geo.loops[outer_loop_idx]
        vs = lp.verts
        if len(vs) < 3:
            return Point3(0.0, 0.0, 1.0)

        p0 = geo.points[vs[0]]
        i1 = 1
        while i1 < len(vs) and vs[i1] == vs[0]:
            i1 += 1
        if i1 >= len(vs):
            return Point3(0.0, 0.0, 1.0)

        i2 = i1 + 1
        while i2 < len(vs) and (vs[i2] == vs[0] or vs[i2] == vs[i1]):
            i2 += 1
        if i2 >= len(vs):
            return Point3(0.0, 0.0, 1.0)

        p1 = geo.points[vs[i1]]
        p2 = geo.points[vs[i2]]

        ux = p1.x - p0.x
        uy = p1.y - p0.y
        uz = p1.z - p0.z

        vx = p2.x - p0.x
        vy = p2.y - p0.y
        vz = p2.z - p0.z

        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        nn = (nx * nx + ny * ny + nz * nz) ** 0.5
        if nn == 0.0:
            return Point3(0.0, 0.0, 1.0)

        inv = 1.0 / nn
        return Point3(nx * inv, ny * inv, nz * inv)

    def _extract_face(
            self,
            geo: SketchUpGeo,
            face: SU.SUFaceRef,
            point_map: dict[tuple[int, int, int], int],
            edge_map: dict[tuple[int, int], int],
    ) -> None:
        outer = SU.SULoopRef()
        SU.su_check(self.su.SUFaceGetOuterLoop(face, byref(outer)), "SUFaceGetOuterLoop")
        outer_idx = self._extract_loop(geo, outer, True, point_map, edge_map)

        inner_indices: list[int] = []
        if SU.has(self.su, "SUFaceGetNumInnerLoops") and SU.has(self.su, "SUFaceGetInnerLoops"):
            inn = c_size_t(0)
            SU.su_check(self.su.SUFaceGetNumInnerLoops(face, byref(inn)), "SUFaceGetNumInnerLoops")
            cnt = int(inn.value)
            if cnt > 0:
                loops = (SU.SULoopRef * cnt)()
                got = c_size_t(0)
                SU.su_check(self.su.SUFaceGetInnerLoops(face, cnt, loops, byref(got)), "SUFaceGetInnerLoops")
                n = int(got.value)
                for i in range(n):
                    inner_indices.append(self._extract_loop(geo, loops[i], False, point_map, edge_map))

        fm = -1
        bm = -1
        if SU.has(self.su, "SUFaceGetFrontMaterial"):
            m = SU.SUMaterialRef()
            ok = SU.su_allow_nodata(self.su.SUFaceGetFrontMaterial(face, byref(m)), "SUFaceGetFrontMaterial")
            if ok:
                fm = self._material_to_index(geo, m)

        if SU.has(self.su, "SUFaceGetBackMaterial"):
            m = SU.SUMaterialRef()
            ok = SU.su_allow_nodata(self.su.SUFaceGetBackMaterial(face, byref(m)), "SUFaceGetBackMaterial")
            if ok:
                bm = self._material_to_index(geo, m)

        nrm = self._face_normal_from_outer_loop(geo, outer_idx)

        geo.faces.append(Face(
            outer_loop=outer_idx,
            inner_loops=inner_indices,
            front_material=fm,
            back_material=bm,
            n=nrm,
        ))

    def _extract_faces_in_entities(
            self,
            geo: SketchUpGeo,
            entities: SU.SUEntitiesRef,
            point_map: dict[tuple[int, int, int], int],
            edge_map: dict[tuple[int, int], int],
    ) -> None:
        fn = c_size_t(0)
        SU.su_check(self.su.SUEntitiesGetNumFaces(entities, byref(fn)), "SUEntitiesGetNumFaces")
        cnt = int(fn.value)
        if cnt <= 0:
            return

        faces = (SU.SUFaceRef * cnt)()
        got = c_size_t(0)
        SU.su_check(self.su.SUEntitiesGetFaces(entities, cnt, faces, byref(got)), "SUEntitiesGetFaces")
        n = int(got.value)
        for i in range(n):
            self._extract_face(geo, faces[i], point_map, edge_map)

    def _walk_entities_recursive(
            self,
            geo: SketchUpGeo,
            entities: SU.SUEntitiesRef,
            depth: int,
            max_depth: int,
            point_map: dict[tuple[int, int, int], int],
            edge_map: dict[tuple[int, int], int],
    ) -> None:
        if depth > max_depth:
            raise RuntimeError(f"Too deep nesting(>{max_depth})")

        self._extract_faces_in_entities(geo, entities, point_map, edge_map)

        n = c_size_t(0)
        SU.su_check(self.su.SUEntitiesGetNumInstances(entities, byref(n)), "SUEntitiesGetNumInstances")
        cnt = int(n.value)
        if cnt <= 0:
            return

        arr = (SU.SUComponentInstanceRef * cnt)()
        got = c_size_t(0)
        SU.su_check(self.su.SUEntitiesGetInstances(entities, cnt, arr, byref(got)), "SUEntitiesGetInstances")
        m = int(got.value)

        for i in range(m):
            definition = SU.SUComponentDefinitionRef()
            SU.su_check(self.su.SUComponentInstanceGetDefinition(arr[i], byref(definition)),
                        "SUComponentInstanceGetDefinition")
            def_entities = SU.SUEntitiesRef()
            SU.su_check(self.su.SUComponentDefinitionGetEntities(definition, byref(def_entities)),
                        "SUComponentDefinitionGetEntities")
            self._walk_entities_recursive(geo, def_entities, depth + 1, max_depth, point_map, edge_map)

    def read(self, skp_path: str, max_depth: int = 64) -> SketchUpGeo:
        skp_path = os.path.abspath(skp_path)
        if not os.path.isfile(skp_path):
            raise FileNotFoundError(skp_path)

        geo = SketchUpGeo(file_path=skp_path)

        geo.su_initialize_result = int(self.su.SUInitialize())

        model = SU.SUModelRef()
        status = c_int(0)

        if SU.has(self.su, "SUModelCreateFromFileWithStatus"):
            geo.load_result = int(
                self.su.SUModelCreateFromFileWithStatus(byref(model), skp_path.encode("utf-8"), byref(status)))
            geo.load_status = int(status.value)
            SU.su_check(geo.load_result, "SUModelCreateFromFileWithStatus")
        else:
            geo.load_result = int(self.su.SUModelCreateFromFile(byref(model), skp_path.encode("utf-8")))
            geo.load_status = 0
            SU.su_check(geo.load_result, "SUModelCreateFromFile")

        geo.model_ptr = int(model.ptr) if model.ptr else 0
        geo.model_name = self._get_model_name(model)
        geo.units = self._get_units(model)

        root = SU.SUEntitiesRef()
        SU.su_check(self.su.SUModelGetEntities(model, byref(root)), "SUModelGetEntities")

        point_map: dict[tuple[int, int, int], int] = {}
        edge_map: dict[tuple[int, int], int] = {}
        self._walk_entities_recursive(geo, root, depth=0, max_depth=max_depth, point_map=point_map, edge_map=edge_map)

        SU.su_check(int(self.su.SUModelRelease(byref(model))), "SUModelRelease")
        int(self.su.SUTerminate())
        return geo


def read_su_mesh(dll_path: str, skp_path: str, max_depth: int = 64, verbose: bool = True) -> SketchUpGeo:
    r = SketchUpReader(dll_path)
    geo = r.read(skp_path, max_depth=max_depth)
    r.close()

    if verbose:
        print(format_mesh_report(geo, width=84, show_materials=10))
    return geo


def step_m_to_model_units(step_m: float, geo_units: str) -> float:
    if geo_units in _UNIT_TO_M:
        return step_m / _UNIT_TO_M[geo_units]
    return step_m


class mainT:
    @staticmethod
    def _main1() -> None:
        in_dll = r"D:\Program Files\SketchUp\SketchUp2025\SketchUp\SketchUpAPI.dll"
        in_skp = r"C:\Users\L\Desktop\无标题.skp"
        _ = read_su_mesh(in_dll, in_skp, max_depth=64, verbose=True)


if __name__ == "__main__":
    mainT._main1()
