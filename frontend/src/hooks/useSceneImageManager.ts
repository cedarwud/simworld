import { useState, useEffect, useRef, useCallback } from 'react';
import { ApiRoutes } from '../config/apiRoutes';
import { getBackendSceneName, getSceneTextureName } from '../utils/sceneUtils';

const FALLBACK_IMAGE_PATH = '/rendered_images/scene_with_devices.png';

export function useSceneImageManager(sceneName?: string) {
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const prevImageUrlRef = useRef<string | null>(null);
    const [usingFallback, setUsingFallback] = useState<boolean>(false);
    const [retryCount, setRetryCount] = useState<number>(0);
    const [manualRetryMode, setManualRetryMode] = useState<boolean>(false);
    const [imageNaturalSize, setImageNaturalSize] = useState<{
        width: number;
        height: number;
    } | null>(null);

    const imageRefToAttach = useRef<HTMLImageElement>(null);

    const fetchImage = useCallback(
        async (signal: AbortSignal) => {
            // Use scene-specific texture if sceneName is provided
            const rtEndpoint = sceneName 
                ? ApiRoutes.scenes.getSceneTexture(getBackendSceneName(sceneName), getSceneTextureName(sceneName))
                : ApiRoutes.simulations.getSceneImage;
            
            setIsLoading(true);
            setError(null);
            setUsingFallback(false);

            if (prevImageUrlRef.current) {
                URL.revokeObjectURL(prevImageUrlRef.current);
                prevImageUrlRef.current = null;
            }

            const endpointWithCacheBuster = `${rtEndpoint}?t=${new Date().getTime()}`;
            let timeoutId: number | null = null;

            try {
                timeoutId = window.setTimeout(() => {
                    // console.warn(`Fetch image request timed out after 15s for ${endpointWithCacheBuster}`);
                }, 15000);

                const response = await fetch(endpointWithCacheBuster, {
                    signal,
                    cache: 'no-cache',
                    headers: { Pragma: 'no-cache', 'Cache-Control': 'no-cache' },
                });

                if (timeoutId !== null) window.clearTimeout(timeoutId);

                if (!response.ok) {
                    let errorDetail = `HTTP error! status: ${response.status}`;
                    try {
                        const errorJson = await response.json();
                        errorDetail = errorJson.detail || errorDetail;
                    } catch (jsonError) { /* Keep original HTTP error */ }
                    throw new Error(errorDetail);
                }

                try {
                    const imageBlob = await response.blob();
                    if (imageBlob.size === 0) {
                        throw new Error('Received empty image blob.');
                    }
                    const newImageUrl = URL.createObjectURL(imageBlob);
                    setImageUrl(newImageUrl);
                    prevImageUrlRef.current = newImageUrl;
                    setRetryCount(0);
                    setManualRetryMode(false);
                } catch (blobError) {
                    throw new Error(
                        `處理圖像時出錯: ${blobError instanceof Error ? blobError.message : String(blobError)}`
                    );
                }
            } catch (error: any) {
                console.error('Error fetching image:', error);

                if (timeoutId !== null) window.clearTimeout(timeoutId);

                if (!signal.aborted) {
                    setError('Error loading image: ' + error.message);
                    setRetryCount((prev) => {
                        const newCount = prev + 1;
                        // If we've tried several times, use fallback
                        if (newCount > 3) {
                            setUsingFallback(true);
                            setImageUrl(FALLBACK_IMAGE_PATH);
                            setManualRetryMode(true);
                            setIsLoading(false);
                            return newCount;
                        }
                        return newCount;
                    });
                }
            } finally {
                if (timeoutId !== null) window.clearTimeout(timeoutId);
                setIsLoading(false);
            }
        },
        [sceneName]
    );

    const handleImageLoad = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
        const img = event.target as HTMLImageElement;
        if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
            setImageNaturalSize({
                width: img.naturalWidth,
                height: img.naturalHeight
            });
        }
        setIsLoading(false);
    }, []);

    const handleImageError = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
        console.error('Image failed to load', event);
        // Only activate fallback after true error (not immediately)
        setUsingFallback(true);
        setImageUrl(FALLBACK_IMAGE_PATH);
        setIsLoading(false);
        setError('無法載入場景圖像，請檢查網絡或重試');
    }, []);

    // Load initial image
    useEffect(() => {
        const abortController = new AbortController();
        fetchImage(abortController.signal);
        return () => {
            abortController.abort();
            if (prevImageUrlRef.current) {
                URL.revokeObjectURL(prevImageUrlRef.current);
                prevImageUrlRef.current = null;
            }
        };
    }, [fetchImage]);

    return {
        imageUrl,
        imageRefToAttach,
        isLoading,
        error,
        usingFallback,
        manualRetryMode,
        imageNaturalSize,
        retryLoad: () => fetchImage(new AbortController().signal),
        handleImageLoad,
        handleImageError,
    };
} 