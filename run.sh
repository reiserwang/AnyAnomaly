#!/bin/bash
# Start Backend
echo "Starting Backend..."
cd backend
source venv/bin/activate
python app.py &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm run dev -- --host &
FRONTEND_PID=$!
cd ..

echo "AnyAnomaly started!"
echo "Backend: http://localhost:5001"
echo "Frontend: http://localhost:5173"
echo "Press CTRL+C to stop."

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

wait
