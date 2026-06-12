import { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCcw, ChevronLeft, Sparkles } from 'lucide-react';
import { clsx } from 'clsx';
import Storyline from './Storyline';
import Timeline from './Timeline';

function AnalysisResult({ videoSrc, data, onReset }) {
    const videoRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);

    // Derived state for overlays
    const currentIndex = Math.min(Math.floor((currentTime / (duration || 1)) * (data.results?.length || 1)), (data.results?.length || 100) - 1);
    const currentScore = data.results ? data.results[currentIndex] : 0;

    // Find active bounding box (from closest storyline item within 1s window)
    const activeStoryItem = findActiveStoryItem(data.storyline, currentTime);
    const activeBox = activeStoryItem?.box; // [ymin, xmin, ymax, xmax] normalized

    const scores = data.results || [];
    const isSummary = data.prompt === 'Video Summary';

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleTimeUpdate = () => setCurrentTime(video.currentTime);
        const handleLoadedMetadata = () => setDuration(video.duration);
        const handleEnded = () => setIsPlaying(false);

        video.addEventListener('timeupdate', handleTimeUpdate);
        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        video.addEventListener('ended', handleEnded);

        return () => {
            video.removeEventListener('timeupdate', handleTimeUpdate);
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
            video.removeEventListener('ended', handleEnded);
        };
    }, []);

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) videoRef.current.pause();
            else videoRef.current.play();
            setIsPlaying(!isPlaying);
        }
    };

    const seek = (time) => {
        if (videoRef.current) {
            videoRef.current.currentTime = time;
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <button
                    onClick={onReset}
                    className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-slate-500 hover:text-white transition-colors"
                >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    New Scan
                </button>

                <div className="label-tech">
                    {isSummary ? 'mode:' : 'target:'}{' '}
                    <span className={clsx("font-semibold", isSummary ? "text-ok" : "text-signal-soft")}>
                        {data.prompt}
                    </span>
                </div>
            </div>

            {/* Summary Section */}
            {data.summary && (
                <div className="relative border border-edge bg-panel/70 p-6">
                    <div className="hud-corners" />
                    <h3 className="label-tech text-ok mb-3 flex items-center gap-2">
                        <Sparkles className="w-3.5 h-3.5" />
                        // video summary
                    </h3>
                    <p className="text-slate-300 leading-relaxed text-sm">
                        {data.summary}
                    </p>
                </div>
            )}

            <div className="relative group bg-black border border-edge">
                <div className="hud-corners z-40" />
                <video
                    ref={videoRef}
                    src={videoSrc}
                    className="w-full h-auto max-h-[60vh] object-contain mx-auto"
                    onClick={togglePlay}
                />

                {/* Play Overlay */}
                {!isPlaying && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-colors pointer-events-none">
                        <div className="p-4 bg-white/10 backdrop-blur-md rounded-full ring-1 ring-white/20">
                            <Play className="w-8 h-8 text-white fill-current" />
                        </div>
                    </div>
                )}

                {/* Temporal Overlay (signal border while score is high) */}
                <div
                    className={clsx(
                        "absolute inset-0 pointer-events-none transition-all duration-300 border-[4px] z-20",
                        currentScore > 0.6 ? "border-signal/80 shadow-[inset_0_0_50px_rgba(99,102,241,0.4)]" : "border-transparent"
                    )}
                />

                {/* Spatial Overlay (Red Bounding Box) */}
                {activeBox && (
                    <div
                        className="absolute z-30 border-2 border-alert bg-alert/10 shadow-[0_0_15px_rgba(244,63,94,0.6)] animate-pulse"
                        style={{
                            top: `${activeBox[0] * 100}%`,
                            left: `${activeBox[1] * 100}%`,
                            height: `${(activeBox[2] - activeBox[0]) * 100}%`,
                            width: `${(activeBox[3] - activeBox[1]) * 100}%`,
                        }}
                    >
                        <div className="absolute -top-5 left-0 bg-alert text-white font-mono text-[9px] font-semibold uppercase tracking-[0.2em] px-1.5 py-0.5">
                            anomaly
                        </div>
                    </div>
                )}
            </div>

            <div className="border border-edge bg-panel/70 p-6">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex gap-2">
                        <button onClick={togglePlay} className="p-2 border border-edge hover:border-signal/60 transition-colors text-slate-300 hover:text-white">
                            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
                        </button>
                        <button onClick={() => seek(0)} className="p-2 border border-edge hover:border-signal/60 transition-colors text-slate-300 hover:text-white">
                            <RotateCcw className="w-4 h-4" />
                        </button>
                    </div>
                    <div className="flex items-center gap-4 font-mono text-xs tabular-nums">
                        {!isSummary && (
                            <span className={clsx(
                                "px-2 py-0.5 border",
                                currentScore > 0.6
                                    ? "border-alert/50 text-alert"
                                    : "border-edge text-slate-500"
                            )}>
                                score {(currentScore ?? 0).toFixed(2)}
                            </span>
                        )}
                        <span className="text-slate-400">
                            {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
                        </span>
                    </div>
                </div>

                {/* Timeline Graph */}
                <Timeline
                    duration={duration}
                    scores={scores}
                    storyline={data.storyline || []}
                    onSeek={seek}
                    currentTime={currentTime}
                />
            </div>

            {/* Storyline Section */}
            {data.storyline && (
                <Storyline
                    scenes={data.storyline}
                    onSceneClick={(timestamp) => seek(timestamp)}
                />
            )}
        </div>
    );
}

function findActiveStoryItem(items, currentTime) {
    if (!items || items.length === 0) return undefined;

    // Binary search: smallest index where items[i].timestamp > currentTime - 1.0,
    // then check it's within the +1.0s upper bound.
    const target = currentTime - 1.0;
    let low = 0;
    let high = items.length - 1;
    let resultIdx = -1;

    while (low <= high) {
        const mid = (low + high) >>> 1;
        if (items[mid].timestamp > target) {
            resultIdx = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    if (resultIdx !== -1) {
        const item = items[resultIdx];
        if (item.timestamp < currentTime + 1.0) {
            return item;
        }
    }
    return undefined;
}

export default AnalysisResult;
