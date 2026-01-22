import { useState } from 'react';
import VideoUpload from './components/VideoUpload';
import AnalysisResult from './components/AnalysisResult';
import { Activity, ShieldCheck } from 'lucide-react';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [progress, setProgress] = useState({ percent: 0, message: '' });
  const [logs, setLogs] = useState([]);

  const handleAnalyze = async (inputData) => {
    setIsAnalyzing(true);
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
      const response = await fetch('http://localhost:5001/analyze', {
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
                setVideoFile(`http://localhost:5001/uploads/${msg.data.filename}`);
              }
            } else if (msg.type === 'error') {
              alert('Error from server: ' + msg.message);
            }
          } catch (e) {
            console.error("Error parsing JSON line", e);
          }
        }
      }

    } catch (error) {
      alert('Error connecting to server: ' + error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-900/20 via-slate-950 to-slate-950 pointer-events-none" />

      <div className="relative max-w-6xl mx-auto px-6 py-12">
        <header className="mb-16 text-center space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-2xl ring-1 ring-indigo-500/20 mb-4 shadow-[0_0_15px_-3px_rgba(99,102,241,0.3)]">
            <Activity className="w-8 h-8 text-indigo-400 mr-3" />
            <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-indigo-200 via-white to-slate-200 bg-clip-text text-transparent">
              C-VAD Core
            </h1>
          </div>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Customizable Video Anomaly Detection powered by MiniCPM-V.
            Describe an event, and we'll find it in your video.
          </p>
        </header>

        <main className="space-y-12">
          {!analysisData && !isAnalyzing && (
            <div className="max-w-xl mx-auto backdrop-blur-sm bg-slate-900/50 p-8 rounded-3xl border border-slate-800 shadow-2xl transition-all duration-500 hover:shadow-indigo-500/10">
              <VideoUpload onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
            </div>
          )}

          {isAnalyzing && (
            <div className="max-w-xl mx-auto p-6 bg-slate-900/80 rounded-2xl border border-slate-800 text-center animate-pulse-fade text-white">
              <h3 className="text-xl font-bold mb-4">Analyzing Video...</h3>

              {/* Progress Bar */}
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden mb-2">
                <div
                  className="h-full bg-indigo-500 transition-all duration-300 ease-out"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
              <p className="text-slate-400 text-sm mb-6">{progress.message}</p>

              {/* Logs Console */}
              <div className="bg-black/50 rounded-lg p-3 h-48 overflow-y-auto text-left font-mono text-xs text-slate-300 border border-white/5 shadow-inner">
                {logs.map((log, i) => (
                  <div key={i} className="mb-1 opacity-80 border-b border-white/5 pb-0.5 last:border-0">
                    <span className="text-indigo-400 mr-2">➜</span>
                    {log}
                  </div>
                ))}
                <div ref={el => el?.scrollIntoView({ behavior: "smooth" })} />
              </div>
            </div>
          )}

          {analysisData && (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700">
              <AnalysisResult
                videoSrc={videoFile}
                data={analysisData}
                onReset={() => { setAnalysisData(null); setVideoFile(null); }}
              />
            </div>
          )}
        </main>

        <footer className="mt-20 text-center text-slate-600 text-sm">
          <p className="flex items-center justify-center gap-2">
            <ShieldCheck className="w-4 h-4" />
            Powered by Paper-AnyAnomaly & MiniCPM-V-2.6
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
