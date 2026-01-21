import React from 'react';

const Storyline = ({ data, duration }) => {
  if (!data || data.length === 0) return null;

  return (
    <div className="w-full mt-8">
      <h2 className="text-2xl font-bold mb-4 text-gray-100">Anomaly Storyline</h2>
      <div className="space-y-8">
        {data.map((item, index) => (
          <div key={item.id} className="bg-slate-800 rounded-lg p-4 shadow-lg border border-slate-700">
            <div className="flex flex-col md:flex-row gap-6">
              
              {/* Visual Frame & Spatial Matches (Red) */}
              <div className="relative shrink-0">
                <div className="relative w-full max-w-[400px] aspect-video bg-black rounded overflow-hidden">
                  <img 
                    src={item.frame_url} 
                    alt={item.description}
                    className="w-full h-full object-cover"
                  />
                  {/* Spatial Matches - RED */}
                  {item.spatial_matches && item.spatial_matches.map((box, i) => (
                    <div
                      key={i}
                      className="absolute border-2 border-red-500 bg-red-500/20"
                      style={{
                        left: `${box.x}px`,
                        top: `${box.y}px`,
                        width: `${box.w}px`,
                        height: `${box.h}px`,
                      }}
                    >
                      <span className="absolute -top-6 left-0 bg-red-500 text-white text-xs px-1 rounded">
                        Match {i+1}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-2 text-sm text-gray-400">
                    Timestamp: {item.timestamp}s
                </div>
              </div>

              {/* Temporal Context & Timeline (Blue) */}
              <div className="flex-1 flex flex-col justify-center">
                <h3 className="text-xl font-semibold text-blue-200 mb-2">{item.description}</h3>
                <p className="text-gray-400 mb-4">
                  Detected at {item.timestamp}s within segment {item.temporal_segment.start}s - {item.temporal_segment.end}s
                </p>

                <div className="w-full bg-slate-900 h-8 rounded-full relative overflow-hidden mt-2 border border-slate-700">
                   {/* Background track */}
                   <div className="absolute inset-0 flex items-center px-2">
                        <span className="text-xs text-gray-600">0s</span>
                        <div className="flex-1"></div>
                        <span className="text-xs text-gray-600">{duration}s</span>
                   </div>

                   {/* Temporal Match - BLUE */}
                   <div 
                        className="absolute top-0 bottom-0 bg-blue-500/80 backdrop-blur-sm transition-all duration-500 flex items-center justify-center"
                        style={{
                            left: `${(item.temporal_segment.start / duration) * 100}%`,
                            width: `${((item.temporal_segment.end - item.temporal_segment.start) / duration) * 100}%`
                        }}
                   >
                        <span className="text-xs text-white font-bold whitespace-nowrap px-1">
                            {item.temporal_segment.start}-{item.temporal_segment.end}s
                        </span>
                   </div>

                   {/* Current Frame Marker */}
                   <div 
                        className="absolute top-0 bottom-0 w-1 bg-white shadow-glow"
                        style={{
                            left: `${(item.timestamp / duration) * 100}%`
                        }}
                   />
                </div>
                <div className="mt-1 text-xs text-blue-400">
                    Temporal Match (Blue Segment)
                </div>
              </div>

            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Storyline;
