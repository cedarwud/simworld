# backend/app/api/v1/router.py
from fastapi import APIRouter, Response, status, Query
import os
from starlette.responses import FileResponse
from datetime import datetime, timedelta
import random
from typing import List, Optional
from pydantic import BaseModel

# Import new domain API routers
from app.domains.device.api.device_api import router as device_router

# 恢復領域API路由
from app.domains.coordinates.api.coordinate_api import router as coordinates_router
from app.domains.satellite.api.satellite_api import router as satellite_router
from app.domains.simulation.api.simulation_api import router as simulation_router

# 引入 Skyfield 相關庫
from skyfield.api import load, wgs84, EarthSatellite
import numpy as np

# 全局狀態變數，用於調試
SKYFIELD_LOADED = False
SATELLITE_COUNT = 0

# 嘗試加載時間尺度和衛星數據
try:
    print("開始加載 Skyfield 時間尺度和衛星數據...")
    ts = load.timescale(builtin=True)
    print("時間尺度加載成功")

    # 優先使用 Celestrak 的活躍衛星數據
    print("從 Celestrak 下載衛星數據...")
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    satellites = load.tle_file(url)
    print(f"衛星數據下載成功，共 {len(satellites)} 顆衛星")

    # 建立衛星字典，以名稱為鍵
    satellites_dict = {sat.name: sat for sat in satellites}

    # 獲取各衛星類別，用於顯示
    starlink_sats = [sat for sat in satellites if "STARLINK" in sat.name.upper()]
    oneweb_sats = [sat for sat in satellites if "ONEWEB" in sat.name.upper()]
    globalstar_sats = [sat for sat in satellites if "GLOBALSTAR" in sat.name.upper()]
    iridium_sats = [sat for sat in satellites if "IRIDIUM" in sat.name.upper()]
    print(
        f"通信衛星統計: Starlink: {len(starlink_sats)}, OneWeb: {len(oneweb_sats)}, Globalstar: {len(globalstar_sats)}, Iridium: {len(iridium_sats)}"
    )

    SKYFIELD_LOADED = True
    SATELLITE_COUNT = len(satellites)

except Exception as e:
    print(f"錯誤：無法加載 Skyfield 數據: {e}")
    ts = None
    satellites = []
    satellites_dict = {}
    SKYFIELD_LOADED = False
    SATELLITE_COUNT = 0

api_router = APIRouter()

# Register domain API routers
api_router.include_router(device_router, prefix="/devices", tags=["Devices"])
# 恢復領域API路由
api_router.include_router(
    coordinates_router, prefix="/coordinates", tags=["Coordinates"]
)
api_router.include_router(satellite_router, prefix="/satellites", tags=["Satellites"])
api_router.include_router(
    simulation_router, prefix="/simulations", tags=["Simulations"]
)


# 添加模型資源路由
@api_router.get("/sionna/models/{model_name}", tags=["Models"])
async def get_model(model_name: str):
    """提供3D模型文件"""
    # 定義模型文件存儲路徑
    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "static",
    )
    models_dir = os.path.join(static_dir, "models")

    # 獲取對應的模型文件
    model_file = os.path.join(models_dir, f"{model_name}.glb")

    # 檢查文件是否存在
    if not os.path.exists(model_file):
        return Response(
            content=f"模型 {model_name} 不存在", status_code=status.HTTP_404_NOT_FOUND
        )

    # 返回模型文件
    return FileResponse(
        path=model_file, media_type="model/gltf-binary", filename=f"{model_name}.glb"
    )


# 添加場景資源路由
@api_router.get("/scenes/{scene_name}/model", tags=["Scenes"])
async def get_scene_model(scene_name: str):
    """提供3D場景模型文件"""
    # 定義場景文件存儲路徑
    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "static",
    )
    scenes_dir = os.path.join(static_dir, "scene")
    scene_dir = os.path.join(scenes_dir, scene_name)

    # 獲取對應的場景模型文件
    model_file = os.path.join(scene_dir, f"{scene_name}.glb")

    # 檢查文件是否存在
    if not os.path.exists(model_file):
        return Response(
            content=f"場景 {scene_name} 的模型不存在",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    # 返回場景模型文件
    return FileResponse(
        path=model_file, media_type="model/gltf-binary", filename=f"{scene_name}.glb"
    )


# 定義衛星可見性數據模型
class VisibleSatelliteInfo(BaseModel):
    norad_id: str
    name: str
    elevation_deg: float
    azimuth_deg: float
    distance_km: float
    velocity_km_s: float
    visible_for_sec: int
    orbit_altitude_km: float
    magnitude: Optional[float] = None


# 添加臨時的衛星可見性模擬端點
@api_router.get("/satellite-ops/visible_satellites", tags=["Satellites"])
async def get_visible_satellites(
    count: int = Query(10, gt=0, le=100),
    min_elevation_deg: float = Query(0, ge=0, le=90),
):
    """返回基於 24.786667, 120.996944 位置可見的真實衛星數據"""
    print(
        f"API 調用: get_visible_satellites(count={count}, min_elevation_deg={min_elevation_deg})"
    )
    print(f"Skyfield 狀態: 已加載={SKYFIELD_LOADED}, 衛星數量={SATELLITE_COUNT}")

    # 使用台灣新竹附近的固定坐標作為觀測點
    observer_lat = 24.786667
    observer_lon = 120.996944
    print(f"觀測點座標: ({observer_lat}, {observer_lon})")

    if not SKYFIELD_LOADED or ts is None or not satellites:
        # 如果 Skyfield 數據未加載成功，返回模擬數據
        print("使用模擬數據，因為 Skyfield 未成功加載")
        sim_satellites = []
        for i in range(count):
            # 生成隨機衛星數據
            elevation = random.uniform(min_elevation_deg, 90)
            satellite = VisibleSatelliteInfo(
                norad_id=f"SIM-{40000 + i}",
                name=f"SIM-SAT-{1000 + i}",
                elevation_deg=elevation,
                azimuth_deg=random.uniform(0, 360),
                distance_km=random.uniform(500, 2000),
                velocity_km_s=random.uniform(5, 8),
                visible_for_sec=int(random.uniform(300, 1200)),
                orbit_altitude_km=random.uniform(500, 1200),
                magnitude=random.uniform(1, 5),
            )
            sim_satellites.append(satellite)

        # 按仰角從高到低排序
        sim_satellites.sort(key=lambda x: x.elevation_deg, reverse=True)

        return {"satellites": sim_satellites, "status": "simulated"}

    try:
        # 計算真實衛星數據
        print("計算真實衛星數據...")

        # 使用 wgs84 創建觀測點
        observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=0)

        # 獲取當前時間
        now = ts.now()
        print(f"當前時間: {now.utc_datetime()}")

        # 計算所有衛星在觀測點的方位角、仰角和距離
        visible_satellites = []

        # 優先考慮通信衛星
        priority_sats = starlink_sats + oneweb_sats + globalstar_sats + iridium_sats
        other_sats = [sat for sat in satellites if sat not in priority_sats]
        all_sats = priority_sats + other_sats

        print(f"開始計算衛星可見性，共 {len(all_sats)} 顆衛星")
        processed_count = 0
        visible_count = 0

        # 計算每個衛星的可見性
        for sat in all_sats[:500]:  # 限制處理數量，避免超時
            processed_count += 1
            try:
                # 計算方位角、仰角和距離
                difference = sat - observer
                topocentric = difference.at(now)
                alt, az, distance = topocentric.altaz()

                # 檢查衛星是否高於最低仰角
                if alt.degrees >= min_elevation_deg:
                    visible_count += 1
                    # 計算軌道信息
                    geocentric = sat.at(now)
                    subpoint = geocentric.subpoint()

                    # 計算速度（近似值）
                    velocity = np.linalg.norm(geocentric.velocity.km_per_s)

                    # 估計可見時間（粗略計算）
                    visible_for_sec = int(1000 * (alt.degrees / 90.0))  # 粗略估計

                    # 創建衛星信息對象
                    satellite_info = VisibleSatelliteInfo(
                        norad_id=str(sat.model.satnum),
                        name=sat.name,
                        elevation_deg=round(alt.degrees, 2),
                        azimuth_deg=round(az.degrees, 2),
                        distance_km=round(distance.km, 2),
                        velocity_km_s=round(float(velocity), 2),
                        visible_for_sec=visible_for_sec,
                        orbit_altitude_km=round(subpoint.elevation.km, 2),
                        magnitude=round(random.uniform(1, 5), 1),  # 星等是粗略估計
                    )

                    visible_satellites.append(satellite_info)

                    # 如果已經收集了足夠的衛星，停止
                    if len(visible_satellites) >= count:
                        print(f"已找到足夠的衛星: {len(visible_satellites)}")
                        break
            except Exception as e:
                print(f"計算衛星 {sat.name} 位置時出錯: {e}")
                continue

        print(
            f"處理完成: 處理了 {processed_count} 顆衛星，找到 {visible_count} 顆可見衛星"
        )

        # 按仰角從高到低排序
        visible_satellites.sort(key=lambda x: x.elevation_deg, reverse=True)

        # 限制返回的衛星數量（保留這個邏輯，以防實際衛星數量超過請求數量）
        visible_satellites = visible_satellites[:count]

        return {
            "satellites": visible_satellites,
            "status": "real",
            "processed": processed_count,
            "visible": visible_count,
        }

    except Exception as e:
        print(f"計算衛星位置時發生錯誤: {e}")
        # 發生錯誤時返回模擬數據
        sim_satellites = []
        for i in range(count):
            elevation = random.uniform(min_elevation_deg, 90)
            satellite = VisibleSatelliteInfo(
                norad_id=f"SIM-ERROR-{i}",
                name=f"ERROR-SIM-{i}",
                elevation_deg=elevation,
                azimuth_deg=random.uniform(0, 360),
                distance_km=random.uniform(500, 2000),
                velocity_km_s=random.uniform(5, 8),
                visible_for_sec=int(random.uniform(300, 1200)),
                orbit_altitude_km=random.uniform(500, 1200),
                magnitude=random.uniform(1, 5),
            )
            sim_satellites.append(satellite)

        return {"satellites": sim_satellites, "status": "error", "error": str(e)}
