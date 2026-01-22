import { useState, useRef } from 'react';
import { Upload, FileVideo, Sparkles } from 'lucide-react';
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
            onAnalyze({ type: 'file', file, prompt, frameInterval });
        } else if (mode === 'url' && url && prompt) {
            onAnalyze({ type: 'url', url, prompt, frameInterval });
        }
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-8">
            {/* Mode Toggle */}
            <div className="flex justify-center">
                <div className="bg-slate-900/80 p-1 rounded-xl flex gap-1 border border-slate-800">
                    <button
                        type="button"
                        onClick={() => setMode('file')}
                        className={clsx(
                            "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                            mode === 'file'
                                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/25"
                                : "text-slate-400 hover:text-white hover:bg-slate-800"
                        )}
                    >
                        Upload Video
                    </button>
                    <button
                        type="button"
                        onClick={() => setMode('url')}
                        className={clsx(
                            "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                            mode === 'url'
                                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/25"
                                : "text-slate-400 hover:text-white hover:bg-slate-800"
                        )}
                    >
                        YouTube URL
                    </button>
                </div>
            </div>

            {mode === 'file' ? (
                <div
                    className={twMerge(
                        "relative flex flex-col items-center justify-center w-full h-64 rounded-2xl border-2 border-dashed transition-all duration-300 ease-in-out cursor-pointer group",
                        dragActive ? "border-indigo-500 bg-indigo-500/10" : "border-slate-700 bg-slate-900/30 hover:border-slate-500 hover:bg-slate-800/50",
                        file ? "border-indigo-500/50 bg-indigo-500/5" : ""
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
                        <div className="flex flex-col items-center animate-in zoom-in-50 duration-300">
                            <div className="p-4 bg-indigo-500/20 rounded-full mb-3 ring-1 ring-indigo-500/30">
                                <FileVideo className="w-8 h-8 text-indigo-400" />
                            </div>
                            <p className="text-sm font-medium text-slate-200">{file.name}</p>
                            <p className="text-xs text-slate-500 mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center text-center p-4">
                            <div className="p-4 bg-slate-800 rounded-full mb-3 group-hover:scale-110 transition-transform duration-300">
                                <Upload className="w-8 h-8 text-slate-400 group-hover:text-indigo-400" />
                            </div>
                            <p className="text-sm font-medium text-slate-300">Click to upload or drag and drop</p>
                            <p className="text-xs text-slate-500 mt-1">MP4, AVI, MOV (max 100MB)</p>
                        </div>
                    )}
                </div>
            ) : (
                <div className="space-y-4">
                    <label className="block text-sm font-medium text-slate-300 pl-1">
                        Paste YouTube Video URL
                    </label>
                    <div className="relative group">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <FileVideo className="h-5 w-5 text-indigo-500/50 group-focus-within:text-indigo-400 transition-colors" />
                        </div>
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            className="block w-full pl-10 pr-3 py-3 bg-slate-900 border border-slate-700 rounded-xl text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all font-medium"
                            placeholder="https://www.youtube.com/watch?v=..."
                        />
                    </div>
                </div>
            )}

            <div className="space-y-4">
                <label className="block text-sm font-medium text-slate-300 pl-1">
                    Describe the anomaly to detect
                </label>
                <div className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Sparkles className="h-5 w-5 text-indigo-500/50 group-focus-within:text-indigo-400 transition-colors" />
                    </div>
                    <input
                        type="text"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        className="block w-full pl-10 pr-3 py-3 bg-slate-900 border border-slate-700 rounded-xl text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all font-medium"
                        placeholder="e.g. fighting, falling, running"
                    />
                </div>
            </div>

            <div className="space-y-4">
                <div className="flex justify-between items-center text-sm font-medium text-slate-300 px-1">
                    <label>Frame Interval (Speed vs Precision)</label>
                    <span className="text-indigo-400 font-bold">{frameInterval} frames</span>
                </div>
                <div className="relative group">
                    <input
                        type="range"
                        min="1"
                        max="30"
                        step="1"
                        value={frameInterval}
                        onChange={(e) => setFrameInterval(parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 px-1 mt-1 font-mono uppercase">
                        <span>Precise (Slow)</span>
                        <span>Fast (Rough)</span>
                    </div>
                </div>
            </div>

            <div className="flex gap-4">
                <button
                    type="submit"
                    onClick={() => {
                        if (mode === 'file' && file && prompt) {
                            onAnalyze({ type: 'file', file, prompt, frameInterval, mode: 'detection' });
                        } else if (mode === 'url' && url && prompt) {
                            onAnalyze({ type: 'url', url, prompt, frameInterval, mode: 'detection' });
                        }
                    }}
                    disabled={(mode === 'file' && !file) || (mode === 'url' && !url) || !prompt || isAnalyzing}
                    className={twMerge(
                        "flex-1 flex items-center justify-center py-4 px-6 rounded-xl text-sm font-semibold text-white shadow-lg transition-all duration-300",
                        ((mode === 'file' && !file) || (mode === 'url' && !url) || !prompt || isAnalyzing)
                            ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                            : "bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 hover:shadow-indigo-500/25 active:scale-[0.98]"
                    )}
                >
                    {isAnalyzing ? (
                        <span className="flex items-center gap-2">
                            <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Analyzing...
                        </span>
                    ) : (
                        "Start Detection"
                    )}
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
                    disabled={(mode === 'file' && !file) || (mode === 'url' && !url) || isAnalyzing}
                    className={twMerge(
                        "flex-1 flex items-center justify-center py-4 px-6 rounded-xl text-sm font-semibold text-white shadow-lg transition-all duration-300",
                        ((mode === 'file' && !file) || (mode === 'url' && !url) || isAnalyzing)
                            ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                            : "bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 hover:shadow-emerald-500/25 active:scale-[0.98]"
                    )}
                >
                    Summarize Video
                </button>
            </div>
        </form>
    );
}

export default VideoUpload;
