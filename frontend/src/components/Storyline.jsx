import { AlertTriangle } from 'lucide-react';
import { clsx } from 'clsx';

function Storyline({ scenes, onSceneClick }) {
    if (!scenes || scenes.length === 0) return null;

    return (
        <div className="space-y-4 pt-4">
            <div className="flex items-baseline gap-4">
                <h3 className="label-tech text-signal-soft whitespace-nowrap">// detection storyline</h3>
                <div className="flex-1 border-t border-edge" />
                <span className="label-tech tabular-nums">{scenes.length} frames</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {scenes.map((scene, idx) => (
                    <div
                        key={idx}
                        onClick={() => onSceneClick(scene.timestamp)}
                        className={clsx(
                            "group relative overflow-hidden border cursor-pointer transition-colors duration-300",
                            scene.score > 0.6
                                ? "border-alert/50 bg-alert/5"
                                : "border-edge bg-panel/60 hover:border-signal/50"
                        )}
                    >
                        <div className="aspect-video relative overflow-hidden bg-void">
                            <img
                                src={scene.image}
                                alt={`Scene at ${scene.timestamp.toFixed(1)}s`}
                                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                                loading="lazy"
                            />

                            {/* Score Overlay */}
                            <div className="absolute top-2 right-2 px-1.5 py-0.5 font-mono text-[10px] font-semibold bg-black/70 backdrop-blur-sm border border-white/10 tabular-nums">
                                <span className={clsx(scene.score > 0.6 ? "text-alert" : "text-ok")}>
                                    {(scene.score * 100).toFixed(0)}%
                                </span>
                            </div>

                            {/* Timestamp Overlay */}
                            <div className="absolute bottom-2 left-2 px-1.5 py-0.5 font-mono text-[10px] bg-black/70 backdrop-blur-sm text-slate-300 border border-white/10 tabular-nums">
                                {scene.timestamp.toFixed(1)}s
                            </div>
                        </div>

                        <div className="p-3">
                            <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                                {scene.reason || "Analyzing frame context..."}
                            </p>

                            {scene.score > 0.6 && (
                                <div className="mt-2 flex items-center gap-1.5 font-mono text-[9px] font-semibold text-alert uppercase tracking-[0.2em]">
                                    <AlertTriangle className="w-3 h-3" />
                                    Anomaly Detected
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default Storyline;
