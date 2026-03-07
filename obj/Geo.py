# obj/Geo.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Point3:
    x: float
    y: float
    z: float


@dataclass
class UV:
    u: float
    v: float


@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int = 255


@dataclass
class Edge:
    a: int
    b: int


@dataclass
class Loop:
    is_outer: bool
    edges: List[int]
    verts: List[int] = field(default_factory=list)  # 新增：保留顶点环，后续算法向/UV更方便


@dataclass
class Texture:
    # png可以是文件路径，也可以是你后续导出时写入的相对路径
    png: str
    uv: List[UV] = field(default_factory=list)

    # 贴图放置的几何语义信息，给exportSU或你自己的定位逻辑用
    base_point: Point3 = field(default_factory=lambda: Point3(0.0, 0.0, 0.0))
    width: float = 0.0
    height_v: float = 0.0

    def height(self) -> float:
        return float(self.height_v)


@dataclass
class Material:
    name: str
    ptr: int = 0

    # Material可以指向Color或Texture（二选一或都为空）
    color_index: int = -1
    texture_index: int = -1

    def has_color(self) -> bool:
        return self.color_index >= 0

    def has_texture(self) -> bool:
        return self.texture_index >= 0


@dataclass
class Face:
    outer_loop: int
    inner_loops: List[int]
    front_material: int
    back_material: int

    # 新增：法向（右手系，按outer_loop顶点顺序计算）
    n: Point3 = field(default_factory=lambda: Point3(0.0, 0.0, 1.0))

    # 你说的ColorImp、UVImp这几个列，我按“导出阶段的import句柄/索引”预留在Face上
    # 后续exportSU里创建材质/UV映射后把结果写回即可
    color_imp: int = -1
    uv_imp: int = -1


@dataclass
class SketchUpGeo:
    file_path: str

    points: List[Point3] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    loops: List[Loop] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)

    # 材质体系：Material可引用Color或Texture
    materials: List[Material] = field(default_factory=list)
    colors: List[Color] = field(default_factory=list)
    textures: List[Texture] = field(default_factory=list)

    # 新增：单位
    units: str = ""

    # 额外：统计/调试
    su_initialize_result: int = 0
    load_result: int = 0
    load_status: int = 0
    model_ptr: int = 0
    model_name: str = ""

    # 你要求的列也可以放在Geo级别，我这里再预留一份“全局导入表”位
    # 用不用你自己决定，不影响readSU
    color_imp: List[int] = field(default_factory=list)
    uv_imp: List[int] = field(default_factory=list)