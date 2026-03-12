"""
Qwen3 TTS API - 文本转语音服务

此API提供多种Qwen3 TTS模型的语音合成服务，根据URL路径自动选择对应的模型。

快速开始:
1. 启动API服务器: uvicorn api:app --host 0.0.0.0 --port 8000
2. 访问 http://localhost:8000/ 查看所有可用端点
3. 使用示例:
   - CustomVoice模型: http://localhost:8000/custom-voice/synthesize?text=你好&speaker=Vivian
   - VoiceDesign模型: http://localhost:8000/voice-design/synthesize?text=你好&instruct=用温柔的女声说
   - Base模型: POST到 http://localhost:8000/base/synthesize 需要ref_audio参数

支持的模型:
1. voice-design: Qwen3-TTS-12Hz-1.7B-VoiceDesign - 根据描述进行音色设计
2. custom-voice: Qwen3-TTS-12Hz-1.7B-CustomVoice - 支持9种优质音色，可通过指令控制风格
3. custom-voice-0.6b: Qwen3-TTS-12Hz-0.6B-CustomVoice - 轻量版CustomVoice模型
4. base: Qwen3-TTS-12Hz-1.7B-Base - 基础模型，支持音色克隆
5. base-0.6b: Qwen3-TTS-12Hz-0.6B-Base - 轻量版Base模型

支持的说话人: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
"""

import os
import io
import contextlib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request, Path
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Dict
from qwen_tts import Qwen3TTSModel

# ========== 可选：手动指定 SoX 路径（如果已安装但 Python 找不到）==========
SOX_PATH = r"C:\Program Files (x86)\sox-14-4-2"  # 请根据实际安装路径修改
if os.path.exists(SOX_PATH) and SOX_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + SOX_PATH

# ========== 全局变量，用于在 lifespan 中共享模型 ==========
models: Dict[str, Qwen3TTSModel] = {}

# ========== 模型配置 ==========
MODEL_CONFIGS = {
    "voice-design": {
        "path": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "type": "voice_design",
        "description": "根据用户提供的描述进行音色设计"
    },
    "custom-voice": {
        "path": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "type": "custom_voice",
        "description": "通过用户指令对目标音色进行风格控制；支持 9 种优质音色"
    },
    "custom-voice-0.6b": {
        "path": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "type": "custom_voice",
        "description": "支持 9 种优质音色，涵盖性别、年龄、语言和方言的多种组合"
    },
    "base": {
        "path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "type": "base",
        "description": "基础模型，支持从用户提供的音频输入中实现 3 秒快速音色克隆"
    },
    "base-0.6b": {
        "path": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "type": "base",
        "description": "基础模型，支持从用户提供的音频输入中实现 3 秒快速音色克隆"
    }
}

# ========== 说话人列表 ==========
SPEAKERS = {
    "Vivian": "明亮、略带锐利感的年轻女声。母语：中文",
    "Serena": "温暖柔和的年轻女声。母语：中文",
    "Uncle_Fu": "音色低沉醇厚的成熟男声。母语：中文",
    "Dylan": "清晰自然的北京青年男声。母语：中文（北京方言）",
    "Eric": "活泼、略带沙哑明亮感的成都男声。母语：中文（四川方言）",
    "Ryan": "富有节奏感的动态男声。母语：英语",
    "Aiden": "清晰中频、阳光的美式男声。母语：英语",
    "Ono_Anna": "轻快灵巧的俏皮日语女声。母语：日语",
    "Sohee": "情感丰富的温暖韩语女声。母语：韩语"
}

# ========== 定义请求体模型（用于 POST）==========
class TTSRequest(BaseModel):
    text: str
    speaker: str = "Vivian"
    language: Optional[str] = None
    instruct: Optional[str] = None

class VoiceDesignRequest(BaseModel):
    text: str
    instruct: str
    language: Optional[str] = None

class VoiceCloneRequest(BaseModel):
    text: str
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    language: Optional[str] = None
    x_vector_only_mode: bool = False

# ========== 延迟加载模型函数 ==========
def load_model(model_name: str) -> Qwen3TTSModel:
    """延迟加载模型"""
    if model_name in models:
        return models[model_name]
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"未知模型: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    print(f"正在加载模型: {model_name} ({config['path']})")
    try:
        model = Qwen3TTSModel.from_pretrained(config["path"])
        models[model_name] = model
        print(f"[OK] {model_name} 加载完成")
        return model
    except Exception as e:
        print(f"[ERROR] 加载模型 {model_name} 失败: {e}")
        raise

# ========== lifespan 上下文管理器（替代 on_event）==========
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    print("Qwen3 TTS API 启动中...")
    print("[OK] API 启动完成 (使用延迟加载模型)")
    yield
    # 清理时无需再次声明 global，直接使用即可
    models.clear()
    print("[STOP] 所有模型已释放")

# ========== 创建 FastAPI 应用，传入 lifespan ==========
app = FastAPI(
    title="Qwen3 TTS API",
    description="支持多种模型的文本转语音服务，根据URL路径选择不同模型",
    lifespan=lifespan
)

# ========== 健康检查 ==========
@app.get("/health")
async def health():
    loaded_models = list(models.keys())
    return {
        "status": "ok", 
        "models_loaded": len(models),
        "loaded_models": loaded_models
    }

# ========== 获取说话人列表 ==========
@app.get("/speakers")
async def get_speakers():
    return {
        "speakers": SPEAKERS,
        "total": len(SPEAKERS)
    }

# ========== 获取模型信息 ==========
@app.get("/models")
async def get_models():
    model_info = {}
    for model_name, config in MODEL_CONFIGS.items():
        model_info[model_name] = {
            "path": config["path"],
            "type": config["type"],
            "description": config["description"],
            "loaded": model_name in models
        }
    return {
        "models": model_info,
        "total": len(MODEL_CONFIGS)
    }

# ========== 核心语音合成函数（供 GET 和 POST 共用）==========
async def synthesize_core(text: str, speaker: str) -> Response:
    # 使用默认的 custom-voice 模型（向后兼容）
    model_name = "custom-voice"
    if model_name not in models:
        raise HTTPException(status_code=503, detail=f"模型 {model_name} 尚未加载")
    
    model = models[model_name]

    try:
        # 生成语音
        wav, sr = model.generate_custom_voice(text, speaker=speaker)

        # 处理返回的音频数据（可能是列表或 numpy 数组）
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        # 如果是二维且可能为 (channels, samples)，转置为 (samples, channels)
        if wav.ndim == 2 and wav.shape[0] <= 2 and wav.shape[1] > wav.shape[0]:
            wav = wav.T

        # 写入内存中的 WAV 文件
        buffer = io.BytesIO()
        sf.write(buffer, wav, samplerate=sr, format="WAV")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

# ========== 通用音频响应生成函数 ==========
def generate_audio_response(wav, sr):
    """生成音频响应"""
    # 处理返回的音频数据（可能是列表或 numpy 数组）
    if isinstance(wav, list):
        wav = np.array(wav, dtype=np.float32)
    # 如果是二维且可能为 (channels, samples)，转置为 (samples, channels)
    if wav.ndim == 2 and wav.shape[0] <= 2 and wav.shape[1] > wav.shape[0]:
        wav = wav.T

    # 写入内存中的 WAV 文件
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=sr, format="WAV")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="audio/wav")

# ========== CustomVoice 模型端点 ==========
@app.post("/custom-voice/synthesize")
async def custom_voice_synthesize_post(request: TTSRequest):
    model_name = "custom-voice"
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        wav, sr = model.generate_custom_voice(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            instruct=request.instruct
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.get("/custom-voice/synthesize")
async def custom_voice_synthesize_get(
    text: str = Query(None, description="要合成的文本"),
    tts: str = Query(None, description="别名，与 text 二选一"),
    speaker: str = Query("Vivian", description="说话人名称"),
    language: str = Query(None, description="语言"),
    instruct: str = Query(None, description="指令")
):
    model_name = "custom-voice"
    
    # 优先使用 text，如果为空则使用 tts
    final_text = text or tts
    if not final_text:
        raise HTTPException(status_code=400, detail="缺少 text 或 tts 参数")
    
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        wav, sr = model.generate_custom_voice(
            text=final_text,
            speaker=speaker,
            language=language,
            instruct=instruct
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

# ========== CustomVoice 0.6B 模型端点 ==========
@app.post("/custom-voice-0.6b/synthesize")
async def custom_voice_06b_synthesize_post(request: TTSRequest):
    model_name = "custom-voice-0.6b"
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        # 0.6B 模型不支持 instruct 参数
        wav, sr = model.generate_custom_voice(
            text=request.text,
            speaker=request.speaker,
            language=request.language
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.get("/custom-voice-0.6b/synthesize")
async def custom_voice_06b_synthesize_get(
    text: str = Query(None, description="要合成的文本"),
    tts: str = Query(None, description="别名，与 text 二选一"),
    speaker: str = Query("Vivian", description="说话人名称"),
    language: str = Query(None, description="语言")
):
    model_name = "custom-voice-0.6b"
    
    # 优先使用 text，如果为空则使用 tts
    final_text = text or tts
    if not final_text:
        raise HTTPException(status_code=400, detail="缺少 text 或 tts 参数")
    
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        wav, sr = model.generate_custom_voice(
            text=final_text,
            speaker=speaker,
            language=language
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

# ========== VoiceDesign 模型端点 ==========
@app.post("/voice-design/synthesize")
async def voice_design_synthesize_post(request: VoiceDesignRequest):
    model_name = "voice-design"
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        wav, sr = model.generate_voice_design(
            text=request.text,
            instruct=request.instruct,
            language=request.language
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.get("/voice-design/synthesize")
async def voice_design_synthesize_get(
    text: str = Query(..., description="要合成的文本"),
    instruct: str = Query(..., description="音色设计指令"),
    language: str = Query(None, description="语言")
):
    model_name = "voice-design"
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        wav, sr = model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

# ========== Base 模型端点 ==========
@app.post("/base/synthesize")
async def base_synthesize_post(request: VoiceCloneRequest):
    model_name = "base"
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        if not request.ref_audio:
            raise HTTPException(status_code=400, detail="Base模型需要ref_audio参数")
        
        wav, sr = model.generate_voice_clone(
            text=request.text,
            ref_audio=request.ref_audio,
            ref_text=request.ref_text,
            language=request.language,
            x_vector_only_mode=request.x_vector_only_mode
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

# ========== Base 0.6B 模型端点 ==========
@app.post("/base-0.6b/synthesize")
async def base_06b_synthesize_post(request: VoiceCloneRequest):
    model_name = "base-0.6b"
    try:
        model = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"加载模型失败: {str(e)}")
    
    try:
        if not request.ref_audio:
            raise HTTPException(status_code=400, detail="Base模型需要ref_audio参数")
        
        wav, sr = model.generate_voice_clone(
            text=request.text,
            ref_audio=request.ref_audio,
            ref_text=request.ref_text,
            language=request.language,
            x_vector_only_mode=request.x_vector_only_mode
        )
        return generate_audio_response(wav[0] if isinstance(wav, list) else wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

# ========== 向后兼容的旧端点（使用默认模型）==========
@app.post("/synthesize")
async def synthesize_post(request: TTSRequest):
    return await custom_voice_synthesize_post(request)

@app.get("/synthesize")
async def synthesize_get(
    text: str = Query(None, description="要合成的文本"),
    tts: str = Query(None, description="别名，与 text 二选一"),
    speaker: str = Query("Vivian", description="说话人名称")
):
    return await custom_voice_synthesize_get(text=text, tts=tts, speaker=speaker)

# ========== 根路径信息 ==========
@app.get("/")
async def root():
    return {
        "message": "Qwen3 TTS API 已启动",
        "description": "支持多种模型的文本转语音服务，根据URL路径选择不同模型",
        "available_models": list(MODEL_CONFIGS.keys()),
        "endpoints": {
            "健康检查": "/health",
            "获取说话人列表": "/speakers",
            "获取模型信息": "/models",
            "CustomVoice 1.7B": {
                "POST": "/custom-voice/synthesize",
                "GET": "/custom-voice/synthesize?text=文本&speaker=说话人&language=语言&instruct=指令"
            },
            "CustomVoice 0.6B": {
                "POST": "/custom-voice-0.6b/synthesize",
                "GET": "/custom-voice-0.6b/synthesize?text=文本&speaker=说话人&language=语言"
            },
            "VoiceDesign": {
                "POST": "/voice-design/synthesize",
                "GET": "/voice-design/synthesize?text=文本&instruct=指令&language=语言"
            },
            "Base 1.7B": {
                "POST": "/base/synthesize (需要ref_audio参数)"
            },
            "Base 0.6B": {
                "POST": "/base-0.6b/synthesize (需要ref_audio参数)"
            },
            "向后兼容": {
                "POST": "/synthesize",
                "GET": "/synthesize?text=文本&speaker=说话人"
            }
        },
        "speakers": list(SPEAKERS.keys()),
        "example_requests": {
            "custom_voice": "/custom-voice/synthesize?text=你好&speaker=Vivian",
            "voice_design": "/voice-design/synthesize?text=你好&instruct=用温柔的女声说"
        }
    }
