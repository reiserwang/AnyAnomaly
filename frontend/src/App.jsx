import { useState } from 'react';
import VideoUpload from './components/VideoUpload';
import AnalysisResult from './components/AnalysisResult';
import { AlertTriangle, X } from 'lucide-react';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  const [progress, setProgress] = useState({ percent: 0, message: '' });
  const [logs, setLogs] = useState([]);

  const handleAnalyze = async (inputData) => {
    setIsAnalyzing(true);
    setError(null);
    setProgress({ percent: 0, message: 'Initializing...' });
    setLogs([]);
    setAnalysisData(null);

    const { type, file, url, prompt, frameInterval } = inputData;

    // Reset previous result
    if (type === 'file') {
      setVideoFile(URL.createObjectURL(file));
    } else {
      setVideoFile(null); // Will be set after response
    }

    const formData = new FormData();
    formData.append('prompt', prompt);

    if (type === 'file') {
      formData.append('video', file);
    } else {
      formData.append('youtube_url', url);
    }

    if (frameInterval) {
      formData.append('frame_interval', frameInterval);
    }

    if (inputData.mode) {
      formData.append('mode', inputData.mode);
    }

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`HTTP error! status: ${response.status} - ${errText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            if (msg.type === 'progress') {
              setProgress({ percent: msg.data.progress, message: msg.data.message });
              setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg.data.message}`]);
            } else if (msg.type === 'result') {
              setAnalysisData(msg.data);
              if (msg.data.filename) {
                setVideoFile(`/uploads/${msg.data.filename}`);
              }
            } else if (msg.type === 'error') {
              setError('Server error: ' + msg.message);
            }
          } catch (e) {
            console.error("Error parsing JSON line", e);
          }
        }
      }

    } catch (err) {
      setError('Failed to connect to server: ' + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen">
      <div className="relative max-w-6xl mx-auto px-6 pb-20">

        {/* Status bar */}
        <div className="flex items-center justify-between py-4 border-b border-edge label-tech">
          <span className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${isAnalyzing ? 'bg-alert animate-blink' : 'bg-ok'}`} />
            <span className={isAnalyzing ? 'text-alert' : 'text-ok'}>
              {isAnalyzing ? 'scanning' : 'sys.ready'}
            </span>
          </span>
          <span className="hidden sm:block">c-vad core // minicpm-v 2.6</span>
          <span>open-vocabulary</span>
        </div>

        <header className="pt-16 pb-14 animate-rise">
          <p className="label-tech text-signal-soft mb-4">// video anomaly detection console</p>
          <h1 className="text-5xl sm:text-7xl font-bold uppercase tracking-tight text-white leading-none">
            Any<span className="text-signal">/</span>Anomaly
          </h1>
          <p className="mt-5 max-w-lg text-sm text-slate-400 leading-relaxed">
            Describe an event in plain language — fighting, falling, a truck where it
            shouldn't be — and the model finds it in your footage, frame by frame.
          </p>
        </header>

        {error && (
          <div className="max-w-2xl mb-6 flex items-start gap-3 p-4 border border-alert/40 bg-alert/10 font-mono text-xs text-rose-300">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-alert" />
            <span className="leading-relaxed">{error}</span>
            <button
              onClick={() => setError(null)}
              aria-label="Dismiss error"
              className="ml-auto text-alert hover:text-rose-200 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        <main>
          {!analysisData && !isAnalyzing && (
            <div className="relative max-w-2xl border border-edge bg-panel/70 backdrop-blur-sm p-8 sm:p-10 animate-rise">
              <div className="hud-corners" />
              <VideoUpload onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
            </div>
          )}

          {isAnalyzing && (
            <div className="relative max-w-2xl border border-edge bg-panel/70 backdrop-blur-sm p-8 animate-rise">
              <div className="hud-corners" />

              <div className="flex items-end justify-between mb-6">
                <span className="flex items-center gap-2.5 label-tech text-alert">
                  <span className="w-2 h-2 rounded-full bg-alert animate-blink" />
                  analysis in progress
                </span>
                <span className="font-mono text-3xl text-white tabular-nums leading-none">
                  {String(progress.percent).padStart(2, '0')}<span className="text-slate-600 text-lg">%</span>
                </span>
              </div>

              {/* Progress bar */}
              <div className="relative w-full h-1 bg-edge overflow-hidden mb-3">
                <div
                  className="h-full bg-signal transition-all duration-300 ease-out"
                  style={{ width: `${progress.percent}%` }}
                />
                <div className="absolute inset-y-0 w-1/4 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
              </div>
              <p className="font-mono text-xs text-slate-500 mb-6 truncate">{progress.message}</p>

              {/* Logs console */}
              <div className="bg-void/80 p-3 h-48 overflow-y-auto text-left font-mono text-[11px] text-slate-400 border border-edge">
                {logs.map((log, i) => (
                  <div key={i} className="mb-1 border-b border-edge/40 pb-0.5 last:border-0">
                    <span className="text-signal mr-2">›</span>
                    {log}
                  </div>
                ))}
                <div ref={el => el?.scrollIntoView({ behavior: "smooth" })} />
              </div>
            </div>
          )}

          {analysisData && (
            <div className="animate-rise">
              <AnalysisResult
                videoSrc={videoFile}
                data={analysisData}
                onReset={() => { setAnalysisData(null); setVideoFile(null); }}
              />
            </div>
          )}
        </main>

        <footer className="mt-24 pt-4 border-t border-edge flex items-center justify-between label-tech">
          <span>anyanomaly</span>
          <span>paper-anyanomaly // minicpm-v 2.6</span>
        </footer>
      </div>
    </div>
  );
}

export default App;
