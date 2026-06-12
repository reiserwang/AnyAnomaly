import { useState, useRef } from 'react';
import { Upload, FileVideo, Crosshair } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function VideoUpload({ onAnalyze, isAnalyzing }) {
    const [mode, setMode] = useState('file'); // 'file' | 'url'
    const [file, setFile] = useState(null);
    const [url, setUrl] = useState('');
    const [prompt, setPrompt] = useState('fighting');
    const [frameInterval, setFrameInterval] = useState(5);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    const handleChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (mode === 'file' && file && prompt) {
            onAnalyze({ type: 'file', file, prompt, frameInterval, mode: 'detection' });
        } else if (mode === 'url' && url && prompt) {
            onAnalyze({ type: 'url', url, prompt, frameInterval, mode: 'detection' });
        }
    };

    const detectDisabled = (mode === 'file' && !file) || (mode === 'url' && !url) || !prompt || isAnalyzing;
    const summarizeDisabled = (mode === 'file' && !file) || (mode === 'url' && !url) || isAnalyzing;

    return (
        <form onSubmit={handleSubmit} className="space-y-8">
            {/* Source toggle */}
            <div>
                <span className="label-tech block mb-3">01 / source</span>
                <div className="flex border border-edge font-mono text-[11px] uppercase tracking-[0.2em]">
                    <button
                        type="button"
                        onClick={() => setMode('file')}
                        className={clsx(
                            "flex-1 py-3 transition-colors",
                            mode === 'file'
                                ? "bg-signal text-white"
                                : "text-slate-500 hover:text-white hover:bg-edge/50"
                        )}
                    >
                        File Upload
                    </button>
                    <button
                        type="button"
                        onClick={() => setMode('url')}
                        className={clsx(
                            "flex-1 py-3 border-l border-edge transition-colors",
                            mode === 'url'
                                ? "bg-signal text-white"
                                : "text-slate-500 hover:text-white hover:bg-edge/50"
                        )}
                    >
                        YouTube URL
                    </button>
                </div>
            </div>

            {mode === 'file' ? (
                <div
                    className={twMerge(
                        "relative flex flex-col items-center justify-center w-full h-56 border border-dashed transition-colors duration-300 cursor-pointer group",
                        dragActive ? "border-signal bg-signal/10" : "border-edge bg-void/40 hover:border-slate-600",
                        file ? "border-signal/50 bg-signal/5" : ""
                    )}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => inputRef.current?.click()}
                >
                    <input
                        ref={inputRef}
                        type="file"
                        className="hidden"
                        accept="video/*"
                        onChange={handleChange}
                    />

                    {file ? (
                        <div className="flex flex-col items-center text-center px-4">
                            <FileVideo className="w-7 h-7 text-signal-soft mb-3" />
                            <p className="font-mono text-xs text-slate-200 break-all">{file.name}</p>
                            <p className="font-mono text-[10px] text-slate-500 mt-1.5 uppercase tracking-wider">
                                {(file.size / (1024 * 1024)).toFixed(2)} MB — loaded
                            </p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center text-center px-4">
                            <Upload className="w-7 h-7 text-slate-500 group-hover:text-signal-soft transition-colors mb-3" />
                            <p className="text-sm text-slate-300">Click to upload or drag and drop</p>
                            <p className="font-mono text-[10px] text-slate-500 mt-1.5 uppercase tracking-wider">mp4 / avi / mov / mkv</p>
                        </div>
                    )}
                </div>
            ) : (
                <div className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                        <FileVideo className="h-4 w-4 text-slate-600 group-focus-within:text-signal-soft transition-colors" />
                    </div>
                    <input
                        type="url"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        className="block w-full pl-10 pr-3 py-3.5 bg-void/60 border border-edge font-mono text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-signal transition-colors"
                        placeholder="https://www.youtube.com/watch?v=..."
                    />
                </div>
            )}

            {/* Target prompt */}
            <div>
                <span className="label-tech block mb-3">02 / target</span>
                <div className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                        <Crosshair className="h-4 w-4 text-slate-600 group-focus-within:text-signal-soft transition-colors" />
                    </div>
                    <input
                        type="text"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        className="block w-full pl-10 pr-3 py-3.5 bg-void/60 border border-edge font-mono text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-signal transition-colors"
                        placeholder="e.g. fighting, falling, running"
                    />
                </div>
            </div>

            {/* Frame interval */}
            <div>
                <div className="flex justify-between items-baseline mb-3">
                    <span className="label-tech">03 / frame interval</span>
                    <span className="font-mono text-sm text-signal-soft tabular-nums">
                        {String(frameInterval).padStart(2, '0')}
                    </span>
                </div>
                <input
                    type="range"
                    min="1"
                    max="30"
                    step="1"
                    value={frameInterval}
                    onChange={(e) => setFrameInterval(parseInt(e.target.value))}
                    className="w-full h-1 bg-edge appearance-none cursor-pointer accent-signal"
                />
                <div className="flex justify-between font-mono text-[9px] text-slate-600 mt-2 uppercase tracking-[0.2em]">
                    <span>precise / slow</span>
                    <span>fast / rough</span>
                </div>
            </div>

            <div className="flex gap-3 pt-2">
                <button
                    type="submit"
                    disabled={detectDisabled}
                    className={twMerge(
                        "flex-1 py-4 text-sm font-semibold uppercase tracking-[0.15em] transition-colors",
                        detectDisabled
                            ? "bg-edge/60 text-slate-600 cursor-not-allowed"
                            : "bg-signal hover:bg-signal-soft text-white"
                    )}
                >
                    Start Detection
                </button>

                <button
                    type="button"
                    onClick={() => {
                        if (mode === 'file' && file) {
                            onAnalyze({ type: 'file', file, prompt: 'Summary', frameInterval, mode: 'summarize' });
                        } else if (mode === 'url' && url) {
                            onAnalyze({ type: 'url', url, prompt: 'Summary', frameInterval, mode: 'summarize' });
                        }
                    }}
                    disabled={summarizeDisabled}
                    className={twMerge(
                        "flex-1 py-4 text-sm font-semibold uppercase tracking-[0.15em] border transition-colors",
                        summarizeDisabled
                            ? "border-edge text-slate-600 cursor-not-allowed"
                            : "border-edge text-slate-300 hover:border-ok/60 hover:text-ok"
                    )}
                >
                    Summarize
                </button>
            </div>
        </form>
    );
}

export default VideoUpload;
