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

    // data.results is an array of scores (0-1) per chunk/frame.
    // We'll visualize this as a timeline graph.

    const scores = data.results || [];
    const maxScore = Math.max(...scores, 0.1); // Avoid div by zero

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
            <div className="flex items-center justify-between mb-2">
                <button
                    onClick={onReset}
                    className="text-slate-400 hover:text-white flex items-center gap-2 text-sm font-medium transition-colors"
                >
                    <ChevronLeft className="w-4 h-4" />
                    Back to Upload
                </button>

                <div className="text-xs font-mono text-slate-500">
                    <span className="text-slate-400">{data.prompt === 'Video Summary' ? 'MODE:' : 'DETECTING:'}</span> <span className={clsx("font-bold", data.prompt === 'Video Summary' ? "text-emerald-400" : "text-indigo-400")}>{data.prompt.toUpperCase()}</span>
                </div>
            </div>

            {/* Summary Section */}
            {data.summary && (
                <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-800/50 backdrop-blur-md mb-6">
                    <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-emerald-400" />
                        Video Summary
                    </h3>
                    <p className="text-slate-300 leading-relaxed text-sm">
                        {data.summary}
                    </p>
                </div>
            )}

            <div className="relative group rounded-3xl overflow-hidden bg-black shadow-2xl ring-1 ring-slate-800">
                <video
                    ref={videoRef}
                    src={videoSrc}
                    className="w-full h-auto max-h-[60vh] object-contain mx-auto"
                    onClick={togglePlay}
                />

                {/* Play Overlay */}
                {!isPlaying && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-colors pointer-events-none">
                        <div className="p-4 bg-white/10 backdrop-blur-md rounded-full shadow-lg ring-1 ring-white/20">
                            <Play className="w-8 h-8 text-white fill-current" />
                        </div>
                    </div>
                )}

                {/* Temporal Overlay (Blue Border for High Score) */}
                <div
                    className={clsx(
                        "absolute inset-0 pointer-events-none transition-all duration-300 border-[6px] rounded-3xl z-20",
                        currentScore > 0.6 ? "border-blue-500/80 shadow-[inset_0_0_50px_rgba(59,130,246,0.5)]" : "border-transparent"
                    )}
                />

                {/* Spatial Overlay (Red Bounding Box) */}
                {activeBox && (
                    <div
                        className="absolute z-30 border-[3px] border-red-500 bg-red-500/10 shadow-[0_0_15px_rgba(239,68,68,0.6)] animate-pulse"
                        style={{
                            top: `${activeBox[0] * 100}%`,
                            left: `${activeBox[1] * 100}%`,
                            height: `${(activeBox[2] - activeBox[0]) * 100}%`,
                            width: `${(activeBox[3] - activeBox[1]) * 100}%`,
                        }}
                    >
                        <div className="absolute -top-6 left-0 bg-red-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow">
                            ANOMALY
                        </div>
                    </div>
                )}
            </div>

            <div className="bg-slate-900/80 rounded-2xl p-6 border border-slate-800/50 backdrop-blur-md">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex gap-4">
                        <button onClick={togglePlay} className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-300 hover:text-white">
                            {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current" />}
                        </button>
                        <button onClick={() => seek(0)} className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-300 hover:text-white">
                            <RotateCcw className="w-5 h-5" />
                        </button>
                    </div>
                    <div className="text-sm font-mono text-slate-400">
                        {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
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
            {
                data.storyline && (
                    <Storyline
                        scenes={data.storyline}
                        onSceneClick={(timestamp) => seek(timestamp)}
                    />
                )
            }
        </div >
    );
}

function findActiveStoryItem(items, currentTime) {
    if (!items || items.length === 0) return undefined;

    // Binary search for the first item where timestamp > currentTime - 1.0
    // We want the smallest index `i` such that `items[i].timestamp > currentTime - 1.0`

    // Actually, `Math.abs(diff) < 1.0` is equivalent to `diff > -1.0` AND `diff < 1.0`
    // item.timestamp - currentTime > -1.0  => item.timestamp > currentTime - 1.0
    // item.timestamp - currentTime < 1.0   => item.timestamp < currentTime + 1.0

    const target = currentTime - 1.0;
    let low = 0;
    let high = items.length - 1;
    let resultIdx = -1;

    // Standard lower_bound for value (currentTime - 1.0)
    // We are looking for first element > (currentTime - 1.0)

    while (low <= high) {
        const mid = (low + high) >>> 1;
        if (items[mid].timestamp > target) {
            resultIdx = mid;
            high = mid - 1; // Try to find a smaller index that still satisfies condition
        } else {
            low = mid + 1;
        }
    }

    if (resultIdx !== -1) {
        const item = items[resultIdx];
        // Check upper bound
        if (item.timestamp < currentTime + 1.0) {
            return item;
        }
    }
    return undefined;
}

export default AnalysisResult;
