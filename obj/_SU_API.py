# _SU_API.py
from __future__ import annotations

import ctypes
from ctypes import (
    c_int, c_size_t, c_char_p, c_void_p, c_double, c_bool,
    byref, create_string_buffer, Structure
)

SU_ERROR_NONE = 0
SU_ERROR_NO_DATA = 9

def has(su, name: str) -> bool:
    return hasattr(su, name)


def su_check(res: int, fn_name: str) -> None:
    if int(res) != SU_ERROR_NONE:
        raise RuntimeError(f"{fn_name} failed, SUResult={int(res)}")



def su_allow_nodata(res: int, fn_name: str) -> bool:
    """
    宽松检查：SU_ERROR_NONE返回True；SU_ERROR_NO_DATA返回False；其他错误抛异常。
    专门给SUFaceGetFrontMaterial/BackMaterial这种“可能本来就没数据”的API用。
    """
    r = int(res)
    if r == SU_ERROR_NONE:
        return True
    if r == SU_ERROR_NO_DATA:
        return False
    raise RuntimeError(f"{fn_name} failed, SUResult={r}")

# --------- Refs ---------
class SUModelRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUEntitiesRef(Structure):
    _fields_ = [("ptr", c_void_p)]

class SUEdgeRef(Structure):
    _fields_ = [("ptr", c_void_p)]

class SUFaceRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SULoopRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUVertexRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUComponentInstanceRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUComponentDefinitionRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUMaterialRef(Structure):
    _fields_ = [("ptr", c_void_p)]
class SUGroupRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUTextureRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SULoopInputRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUStringRef(Structure):
    _fields_ = [("ptr", c_void_p)]


class SUPoint3D(Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]


class SUPoint2D(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


class SUVector3D(Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]


class SUMaterialPositionInput(Structure):
    _fields_ = [
        ("num_uv_coords", c_size_t),
        ("uv_coords", SUPoint2D * 4),
        ("points", SUPoint3D * 4),
        ("material", SUMaterialRef),
        ("projection", SUVector3D),
    ]


def invalid_ref(ref_cls):
    r = ref_cls()
    r.ptr = c_void_p(0)
    return r


# --------- UTF8 string helpers ---------
def bind_string_api(su) -> None:
    if not (has(su, "SUStringCreate") and has(su, "SUStringRelease") and has(su, "SUStringGetUTF8Length") and has(su, "SUStringGetUTF8")):
        return

    su.SUStringCreate.argtypes = [ctypes.POINTER(SUStringRef)]
    su.SUStringCreate.restype = c_int

    su.SUStringRelease.argtypes = [ctypes.POINTER(SUStringRef)]
    su.SUStringRelease.restype = c_int

    su.SUStringGetUTF8Length.argtypes = [SUStringRef, ctypes.POINTER(c_size_t)]
    su.SUStringGetUTF8Length.restype = c_int

    su.SUStringGetUTF8.argtypes = [SUStringRef, c_size_t, c_char_p, ctypes.POINTER(c_size_t)]
    su.SUStringGetUTF8.restype = c_int


def su_string_to_py(su, s: SUStringRef) -> str:
    n = c_size_t(0)
    su_check(su.SUStringGetUTF8Length(s, byref(n)), "SUStringGetUTF8Length")
    buf = create_string_buffer(int(n.value) + 1)
    got = c_size_t(0)
    su_check(su.SUStringGetUTF8(s, int(n.value) + 1, buf, byref(got)), "SUStringGetUTF8")
    return buf.value.decode("utf-8", errors="replace")


# --------- Read/inspect bindings ---------
def bind_core_api(su) -> None:
    # lifecycle
    su.SUInitialize.argtypes = []
    su.SUInitialize.restype = c_int

    su.SUTerminate.argtypes = []
    su.SUTerminate.restype = c_int

    # load
    if has(su, "SUModelCreateFromFileWithStatus"):
        su.SUModelCreateFromFileWithStatus.argtypes = [ctypes.POINTER(SUModelRef), c_char_p, ctypes.POINTER(c_int)]
        su.SUModelCreateFromFileWithStatus.restype = c_int
    else:
        su.SUModelCreateFromFile.argtypes = [ctypes.POINTER(SUModelRef), c_char_p]
        su.SUModelCreateFromFile.restype = c_int

    su.SUModelRelease.argtypes = [ctypes.POINTER(SUModelRef)]
    su.SUModelRelease.restype = c_int

    su.SUModelGetEntities.argtypes = [SUModelRef, ctypes.POINTER(SUEntitiesRef)]
    su.SUModelGetEntities.restype = c_int

    # optional: units
    if has(su, "SUModelGetUnits"):
        su.SUModelGetUnits.argtypes = [SUModelRef, ctypes.POINTER(c_int)]
        su.SUModelGetUnits.restype = c_int

    # entities: faces / instances
    su.SUEntitiesGetNumFaces.argtypes = [SUEntitiesRef, ctypes.POINTER(c_size_t)]
    su.SUEntitiesGetNumFaces.restype = c_int
    su.SUEntitiesGetFaces.argtypes = [SUEntitiesRef, c_size_t, ctypes.POINTER(SUFaceRef), ctypes.POINTER(c_size_t)]
    su.SUEntitiesGetFaces.restype = c_int

    su.SUEntitiesGetNumInstances.argtypes = [SUEntitiesRef, ctypes.POINTER(c_size_t)]
    su.SUEntitiesGetNumInstances.restype = c_int
    su.SUEntitiesGetInstances.argtypes = [SUEntitiesRef, c_size_t, ctypes.POINTER(SUComponentInstanceRef), ctypes.POINTER(c_size_t)]
    su.SUEntitiesGetInstances.restype = c_int

    su.SUComponentInstanceGetDefinition.argtypes = [SUComponentInstanceRef, ctypes.POINTER(SUComponentDefinitionRef)]
    su.SUComponentInstanceGetDefinition.restype = c_int
    su.SUComponentDefinitionGetEntities.argtypes = [SUComponentDefinitionRef, ctypes.POINTER(SUEntitiesRef)]
    su.SUComponentDefinitionGetEntities.restype = c_int

    # face/loop/vertex (required)
    if not has(su, "SUFaceGetOuterLoop"):
        raise AttributeError("缺少SUFaceGetOuterLoop，无法提取loop/顶点")
    su.SUFaceGetOuterLoop.argtypes = [SUFaceRef, ctypes.POINTER(SULoopRef)]
    su.SUFaceGetOuterLoop.restype = c_int

    if has(su, "SUFaceGetNumInnerLoops") and has(su, "SUFaceGetInnerLoops"):
        su.SUFaceGetNumInnerLoops.argtypes = [SUFaceRef, ctypes.POINTER(c_size_t)]
        su.SUFaceGetNumInnerLoops.restype = c_int
        su.SUFaceGetInnerLoops.argtypes = [SUFaceRef, c_size_t, ctypes.POINTER(SULoopRef), ctypes.POINTER(c_size_t)]
        su.SUFaceGetInnerLoops.restype = c_int

    if not (has(su, "SULoopGetNumVertices") and has(su, "SULoopGetVertices")):
        raise AttributeError("缺少SULoopGetNumVertices/SULoopGetVertices，无法提取顶点")
    su.SULoopGetNumVertices.argtypes = [SULoopRef, ctypes.POINTER(c_size_t)]
    su.SULoopGetNumVertices.restype = c_int
    su.SULoopGetVertices.argtypes = [SULoopRef, c_size_t, ctypes.POINTER(SUVertexRef), ctypes.POINTER(c_size_t)]
    su.SULoopGetVertices.restype = c_int

    su.SUVertexGetPosition.argtypes = [SUVertexRef, ctypes.POINTER(SUPoint3D)]
    su.SUVertexGetPosition.restype = c_int

    # materials (optional)
    if has(su, "SUFaceGetFrontMaterial"):
        su.SUFaceGetFrontMaterial.argtypes = [SUFaceRef, ctypes.POINTER(SUMaterialRef)]
        su.SUFaceGetFrontMaterial.restype = c_int
    if has(su, "SUFaceGetBackMaterial"):
        su.SUFaceGetBackMaterial.argtypes = [SUFaceRef, ctypes.POINTER(SUMaterialRef)]
        su.SUFaceGetBackMaterial.restype = c_int

    if has(su, "SUMaterialGetName"):
        su.SUMaterialGetName.argtypes = [SUMaterialRef, ctypes.POINTER(SUStringRef)]
        su.SUMaterialGetName.restype = c_int

    # model name (optional)
    if has(su, "SUModelGetName"):
        su.SUModelGetName.argtypes = [SUModelRef, ctypes.POINTER(SUStringRef)]
        su.SUModelGetName.restype = c_int

    # string API
    bind_string_api(su)


# --------- Export/write bindings (新增) ---------
def bind_export_api(su) -> None:
    # create model
    if not has(su, "SUModelCreate"):
        raise AttributeError("缺少SUModelCreate，无法创建新模型")
    su.SUModelCreate.argtypes = [ctypes.POINTER(SUModelRef)]
    su.SUModelCreate.restype = c_int

    # save
    if not has(su, "SUModelSaveToFile"):
        raise AttributeError("缺少SUModelSaveToFile，无法保存skp")
    su.SUModelSaveToFile.argtypes = [SUModelRef, c_char_p]
    su.SUModelSaveToFile.restype = c_int

    # add materials to model
    if not has(su, "SUModelAddMaterials"):
        raise AttributeError("缺少SUModelAddMaterials，无法注册材质")
    su.SUModelAddMaterials.argtypes = [SUModelRef, c_size_t, ctypes.POINTER(SUMaterialRef)]
    su.SUModelAddMaterials.restype = c_int

    # entities add faces
    if not has(su, "SUEntitiesAddFaces"):
        raise AttributeError("缺少SUEntitiesAddFaces，无法添加面")
    su.SUEntitiesAddFaces.argtypes = [SUEntitiesRef, c_size_t, ctypes.POINTER(SUFaceRef)]
    su.SUEntitiesAddFaces.restype = c_int

    # loop input
    if not (has(su, "SULoopInputCreate") and has(su, "SULoopInputAddVertexIndex")):
        raise AttributeError("缺少SULoopInputCreate/SULoopInputAddVertexIndex，无法创建面loop")
    su.SULoopInputCreate.argtypes = [ctypes.POINTER(SULoopInputRef)]
    su.SULoopInputCreate.restype = c_int
    su.SULoopInputAddVertexIndex.argtypes = [SULoopInputRef, c_size_t]
    su.SULoopInputAddVertexIndex.restype = c_int

    # face create
    if not has(su, "SUFaceCreate"):
        raise AttributeError("缺少SUFaceCreate，无法创建SUFace")
    su.SUFaceCreate.argtypes = [ctypes.POINTER(SUFaceRef), ctypes.POINTER(SUPoint3D), ctypes.POINTER(SULoopInputRef)]
    su.SUFaceCreate.restype = c_int

    # face set materials
    if not (has(su, "SUFaceSetFrontMaterial") and has(su, "SUFaceSetBackMaterial")):
        raise AttributeError("缺少SUFaceSetFrontMaterial/SUFaceSetBackMaterial，无法设置正反面材质")
    su.SUFaceSetFrontMaterial.argtypes = [SUFaceRef, SUMaterialRef]
    su.SUFaceSetFrontMaterial.restype = c_int
    su.SUFaceSetBackMaterial.argtypes = [SUFaceRef, SUMaterialRef]
    su.SUFaceSetBackMaterial.restype = c_int

    # position texture on face
    if not has(su, "SUFacePositionMaterial"):
        raise AttributeError("缺少SUFacePositionMaterial，无法定位UV贴图")
    su.SUFacePositionMaterial.argtypes = [SUFaceRef, c_bool, ctypes.POINTER(SUMaterialPositionInput)]
    su.SUFacePositionMaterial.restype = c_int

    # material + texture
    if not (has(su, "SUMaterialCreate") and has(su, "SUMaterialSetName") and has(su, "SUMaterialSetTexture")):
        raise AttributeError("缺少SUMaterialCreate/SUMaterialSetName/SUMaterialSetTexture，无法创建材质并绑定贴图")
    su.SUMaterialCreate.argtypes = [ctypes.POINTER(SUMaterialRef)]
    su.SUMaterialCreate.restype = c_int
    su.SUMaterialSetName.argtypes = [SUMaterialRef, c_char_p]
    su.SUMaterialSetName.restype = c_int
    su.SUMaterialSetTexture.argtypes = [SUMaterialRef, SUTextureRef]
    su.SUMaterialSetTexture.restype = c_int

    if not has(su, "SUTextureCreateFromFile"):
        raise AttributeError("缺少SUTextureCreateFromFile，无法从png创建SUTexture")
    su.SUTextureCreateFromFile.argtypes = [ctypes.POINTER(SUTextureRef), c_char_p, c_double, c_double]
    su.SUTextureCreateFromFile.restype = c_int