import { Clock, AlertTriangle } from 'lucide-react';
import { clsx } from 'clsx';

function Storyline({ scenes, onSceneClick }) {
    if (!scenes || scenes.length === 0) return null;

    // Filter scenes to only show interesting ones or show all? 
    // Plan says "highlight high-anomaly frames".
    // Let's show all but styling differs.

    return (
        <div className="space-y-4">
            <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                <Clock className="w-5 h-5 text-indigo-400" />
                Detection Storyline
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {scenes.map((scene, idx) => (
                    <div
                        key={idx}
                        onClick={() => onSceneClick(scene.timestamp)}
                        className={clsx(
                            "group relative rounded-xl overflow-hidden border cursor-pointer transition-all duration-300 hover:scale-[1.02]",
                            scene.score > 0.6
                                ? "border-red-500/50 bg-red-900/10 shadow-[0_0_15px_rgba(239,68,68,0.2)]"
                                : "border-slate-800 bg-slate-900/50 hover:border-indigo-500/50"
                        )}
                    >
                        <div className="aspect-video relative overflow-hidden bg-slate-950">
                            <img
                                src={`http://localhost:5001${scene.image}`}
                                alt={`Scene at ${scene.timestamp.toFixed(1)}s`}
                                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                                loading="lazy"
                            />

                            {/* Score Overlay */}
                            <div className="absolute top-2 right-2 px-2 py-1 rounded-md text-xs font-bold bg-black/70 backdrop-blur-sm border border-white/10">
                                <span className={clsx(
                                    scene.score > 0.6 ? "text-red-400" : "text-emerald-400"
                                )}>
                                    {(scene.score * 100).toFixed(0)}%
                                </span>
                            </div>

                            {/* Timestamp Overlay */}
                            <div className="absolute bottom-2 left-2 px-2 py-1 rounded-md text-xs font-mono bg-black/70 backdrop-blur-sm text-slate-300 border border-white/10">
                                {scene.timestamp.toFixed(1)}s
                            </div>
                        </div>

                        <div className="p-3">
                            {/* Reason / Caption (Coming from backend reasoning) */}
                            <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                                {scene.reason || "Analyzing frame context..."}
                            </p>

                            {scene.score > 0.6 && (
                                <div className="mt-2 flex items-center gap-1.5 text-[10px] font-bold text-red-400 uppercase tracking-wider">
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
