import React, { useState } from 'react';
import Storyline from './components/Storyline';
import { Search, Loader2, Video } from 'lucide-react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!prompt) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5001/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error analyzing:', error);
      alert("Failed to connect to backend. Make sure the Python server is running on port 5001.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 p-8">
      <div className="max-w-4xl mx-auto">
        
        {/* Header */}
        <header className="flex items-center gap-3 mb-12">
            <div className="p-3 bg-indigo-500 rounded-lg">
                <Video className="w-8 h-8 text-white" />
            </div>
            <div>
                <h1 className="text-3xl font-bold text-white">AnyAnomaly</h1>
                <p className="text-slate-400">Natural Language Video Anomaly Detection</p>
            </div>
        </header>

        {/* Input Section */}
        <div className="bg-slate-900 rounded-xl p-6 shadow-xl border border-slate-800 mb-8">
            <form onSubmit={handleAnalyze} className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
                <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the anomaly (e.g., 'fighting', 'person falling', 'red truck')..."
                    className="w-full bg-slate-950 text-white pl-12 pr-32 py-4 rounded-lg border border-slate-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all placeholder-slate-500"
                />
                <button
                    type="submit"
                    disabled={loading || !prompt}
                    className="absolute right-2 top-2 bottom-2 bg-indigo-600 hover:bg-indigo-700 text-white px-6 rounded-md font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Analyzing...
                        </>
                    ) : (
                        'Analyze'
                    )}
                </button>
            </form>
            <div className="mt-4 flex gap-2">
                <button onClick={() => setPrompt("Person running")} className="text-xs bg-slate-800 text-slate-300 px-3 py-1 rounded-full hover:bg-slate-700 transition-colors">
                    Example: "Person running"
                </button>
                <button onClick={() => setPrompt("Suspicious bag")} className="text-xs bg-slate-800 text-slate-300 px-3 py-1 rounded-full hover:bg-slate-700 transition-colors">
                    Example: "Suspicious bag"
                </button>
            </div>
        </div>

        {/* Results */}
        {result && (
            <Storyline data={result.storyline} duration={result.video_duration} />
        )}
      </div>
    </div>
  );
}

export default App;