import { useMemo } from 'react';
import { clsx } from 'clsx';
import { MapPin, Flag, AlertCircle } from 'lucide-react';

function Timeline({ duration, scores = [], storyline = [], onSeek, currentTime }) {
    const maxScore = Math.max(...scores, 0.1);

    // 1. Calculate Regions (High anomaly spans > 0.6)
    const regions = useMemo(() => {
        const threshold = 0.6;
        const result = [];
        let start = null;

        scores.forEach((score, i) => {
            if (score > threshold && start === null) {
                start = i;
            } else if (score <= threshold && start !== null) {
                result.push({ start, end: i - 1 });
                start = null;
            }
        });
        if (start !== null) result.push({ start, end: scores.length - 1 });
        return result;
    }, [scores]);

    // 2. Calculate Peaks (Local maxima > 0.75 for prominent flags)
    const peaks = useMemo(() => {
        const threshold = 0.75;
        const result = [];
        for (let i = 1; i < scores.length - 1; i++) {
            if (scores[i] > threshold && scores[i] > scores[i - 1] && scores[i] > scores[i + 1]) {
                result.push({ index: i, score: scores[i] });
            }
        }
        return result;
    }, [scores]);

    const getLeftPercent = (index) => (index / (scores.length - 1 || 1)) * 100;
    const getTimestamp = (index) => (index / (scores.length - 1 || 1)) * duration;

    return (
        <div
            className="relative h-32 bg-slate-950/50 rounded-lg select-none border border-slate-800 cursor-pointer group mt-8"
            onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const clickedRatio = x / rect.width;
                onSeek(clickedRatio * duration);
            }}
        >
            {/* Layer 1: Region Highlights (Background) */}
            {regions.map((r, i) => {
                const left = getLeftPercent(r.start);
                const width = getLeftPercent(r.end) - left;
                return (
                    <div
                        key={`region-${i}`}
                        className="absolute top-0 bottom-0 bg-red-900/20 border-x border-red-500/20"
                        style={{ left: `${left}%`, width: `${width}%` }}
                    />
                );
            })}

            {/* Layer 2: Bar Graph (Data) */}
            <div className="absolute bottom-0 left-0 right-0 h-20 flex items-end gap-[1px] opacity-80 px-1">
                {scores.map((score, i) => (
                    <div
                        key={i}
                        className={clsx(
                            "flex-1 rounded-t-[1px] transition-all duration-300",
                            score > 0.6 ? "bg-gradient-to-t from-red-600 to-red-400" :
                                score > 0.4 ? "bg-gradient-to-t from-orange-600 to-orange-400" :
                                    "bg-slate-700"
                        )}
                        style={{ height: `${(score / maxScore) * 100}%` }}
                    />
                ))}
            </div>

            {/* Layer 3: Storyline Pins (Diamonds) */}
            {storyline.map((scene, i) => {
                const percent = (scene.timestamp / duration) * 100;
                return (
                    <div
                        key={`pin-${i}`}
                        className="absolute top-2 -ml-1.5 hover:z-20 group/pin"
                        style={{ left: `${percent}%` }}
                        onClick={(e) => {
                            e.stopPropagation();
                            onSeek(scene.timestamp);
                        }}
                    >
                        <MapPin
                            className={clsx(
                                "w-4 h-4 drop-shadow-lg transition-transform hover:scale-125",
                                scene.score > 0.6 ? "text-red-400 fill-red-900/50" : "text-emerald-400 fill-emerald-900/50"
                            )}
                        />

                        {/* Hover Preview Tooltip */}
                        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 opacity-0 group-hover/pin:opacity-100 transition-opacity pointer-events-none z-50">
                            <div className="bg-slate-900 border border-slate-700 rounded-lg p-2 shadow-xl flex flex-col items-center w-32">
                                <img src={`http://localhost:5001${scene.image}`} className="w-full h-16 object-cover rounded mb-1" />
                                <span className="text-[10px] text-slate-300 font-mono">{scene.timestamp.toFixed(1)}s</span>
                            </div>
                        </div>
                    </div>
                );
            })}

            {/* Layer 4: Peak Markers (Flags with Labels) */}
            {peaks.map((peak, i) => {
                const left = getLeftPercent(peak.index);
                // Only show if significantly high
                if (peak.score < 0.8) return null;

                return (
                    <div
                        key={`peak-${i}`}
                        className="absolute top-0 -ml-px h-full border-l border-dashed border-red-400/30 w-px pointer-events-none"
                        style={{ left: `${left}%` }}
                    >
                        <div className="absolute top-[-24px] -left-3 flex flex-col items-center group/peak pointer-events-auto">
                            <div className="flex items-center gap-1 bg-red-500/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow-lg backdrop-blur hover:bg-red-500 transition-colors">
                                <AlertCircle className="w-3 h-3" />
                                <span>{(peak.score * 100).toFixed(0)}%</span>
                            </div>
                            <div className="h-2 w-px bg-red-500/50"></div>
                        </div>
                    </div>
                );
            })}

            {/* Progress Line */}
            <div
                className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_10px_white] z-10 pointer-events-none transition-all duration-100 ease-linear"
                style={{ left: `${(currentTime / duration) * 100}%` }}
            >
                <div className="absolute top-0 -ml-1 -mt-1 w-2.5 h-2.5 bg-white rounded-full shadow" />
            </div>

            {/* Hover Guide (Optional, could add later) */}
        </div>
    );
}

export default Timeline;
