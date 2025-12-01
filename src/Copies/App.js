import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { initializeApp } from 'firebase/app';
import { getFirestore, doc, setDoc } from 'firebase/firestore';

// --- Utility Components ---

/**
 * Custom Loading Spinner
 */
const LoadingSpinner = ({ message = "Analyzing symptoms..." }) => (
  <div className="flex flex-col items-center justify-center p-6 bg-white rounded-xl shadow-lg">
    <div className="animate-spin rounded-full h-12 w-12 border-4 border-b-4 border-blue-500 border-opacity-50 border-b-blue-700"></div>
    <p className="mt-4 text-gray-600 font-medium text-center">{message}</p>
  </div>
);

/**
 * Emits a calming, professional warning for health applications.
 */
const DisclaimerBanner = () => (
  <div className="p-4 bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 rounded-lg shadow-inner mt-6 max-w-2xl mx-auto">
    <p className="font-semibold text-sm">
      ⚠️ Important Disclaimer
    </p>
    <p className="text-xs mt-1">
      This AI tool provides information, not a medical diagnosis. Always consult a qualified healthcare professional for medical advice, diagnosis, and treatment.
    </p>
  </div>
);

/**
 * Simple icon component for consistency. (Lucide Icons via inline SVG)
 */
const Icon = ({ name, className = 'w-5 h-5' }) => {
  switch (name) {
    case 'microscope':
      return <svg xmlns="http://www.w3.org/2000/svg" className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 18h8M14 14V6M14 8c-2.8 2-5 3.5-5 5a4 4 0 0 0 4 4V8zM8 2h8M10 21l2-2M10 2v2M12 2v2M2 12h2M18 12h2M12 20v2M12 2v2M12 20v2M4 4l2 2M20 4l-2 2M4 20l2-2M20 20l-2-2"/></svg>;
    case 'chat':
      return <svg xmlns="http://www.w3.org/2000/svg" className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8L3 21Z"/></svg>;
    case 'home':
      return <svg xmlns="http://www.w3.org/2000/svg" className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>;
    case 'send':
      return <svg xmlns="http://www.w3.org/2000/svg" className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>;
    case 'error':
      return <svg xmlns="http://www.w3.org/2000/svg" className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>;
    case 'check':
      return <svg xmlns="http://www.w3.org/2000/svg" className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5"/></svg>;
    case 'refresh':
      return <svg xmlns="http://www.w3.org/2000/svg" className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 2v6h6"/><path d="M21 22v-6h-6"/><path d="M21 16a9 9 0 0 0-9-9 9 9 0 0 0-9 9"/></svg>;
    default:
      return null;
  }
};

/**
 * Custom hook to handle Gemini API calls with exponential backoff.
 */
const useGeminiApi = () => {
  const model = "gemini-2.5-flash-preview-05-20";
  const apiKey = "";
  const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const maxRetries = 5;

  const callApi = useCallback(async (userQuery, systemPrompt, useGrounding = true) => {
    const payload = {
      contents: [{ parts: [{ text: userQuery }] }],
      systemInstruction: { parts: [{ text: systemPrompt }] },
      tools: useGrounding ? [{ "google_search": {} }] : undefined,
    };

    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (response.status === 429 && i < maxRetries - 1) {
          const delay = Math.pow(2, i) * 1000 + Math.random() * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }

        if (!response.ok) {
          throw new Error(`API returned status ${response.status}`);
        }

        const result = await response.json();
        const candidate = result.candidates?.[0];

        if (candidate && candidate.content?.parts?.[0]?.text) {
          const text = candidate.content.parts[0].text;
          
          let sources = [];
          const groundingMetadata = candidate.groundingMetadata;
          if (groundingMetadata && groundingMetadata.groundingAttributions) {
              sources = groundingMetadata.groundingAttributions
                  .map(attribution => ({
                      uri: attribution.web?.uri,
                      title: attribution.web?.title,
                  }))
                  .filter(source => source.uri && source.title);
          }

          return { text, sources };
        } else {
          console.error("API response lacked text content:", result);
          throw new Error("Could not generate content from the API.");
        }
      } catch (error) {
        console.error(`Attempt ${i + 1} failed:`, error);
        if (i === maxRetries - 1) throw new Error("API request failed after multiple retries.");
      }
    }
  }, [apiUrl, maxRetries]);

  return callApi;
};



/**
 * Main component for Symptom Checker input.
 */
const SymptomCheckerForm = ({ symptoms, setSymptoms, onSubmit, loading, error }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (symptoms.symptomText.length > 20 && symptoms.age > 0 && symptoms.sex !== '') {
      onSubmit();
    }
  };

  const isFormValid = symptoms.symptomText.length > 20 && symptoms.age > 0 && symptoms.sex !== '';

  return (
    <div className="p-4 sm:p-8 bg-white rounded-xl shadow-2xl w-full max-w-2xl mx-auto">
      <h2 className="text-3xl font-extrabold text-blue-800 mb-2 flex items-center">
        <Icon name="microscope" className="w-8 h-8 mr-2 text-blue-500" />
        Symptom Analyzer
      </h2>
      <p className="text-gray-600 mb-6 border-b pb-4">
        Describe your current symptoms and health context. Be as specific as possible.
      </p>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="symptoms" className="block text-sm font-medium text-gray-700 mb-2">
            What are you feeling today?
          </label>
          <textarea
            id="symptoms"
            rows="4"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 transition duration-150 ease-in-out resize-none"
            placeholder="E.g., I have a sharp headache behind my left eye, a low-grade fever of 100.5°F, and a persistent cough that started 3 days ago."
            value={symptoms.symptomText}
            onChange={(e) => setSymptoms(prev => ({ ...prev, symptomText: e.target.value }))}
            required
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div>
            <label htmlFor="age" className="block text-sm font-medium text-gray-700 mb-2">
              Age (Years)
            </label>
            <input
              id="age"
              type="number"
              min="1"
              max="120"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., 35"
              value={symptoms.age}
              onChange={(e) => setSymptoms(prev => ({ ...prev, age: parseInt(e.target.value) || 0 }))}
              required
            />
          </div>

          <div>
            <label htmlFor="height" className="block text-sm font-medium text-gray-700 mb-2">
              Height (cm)
            </label>
            <input
              id="height"
              type="number"
              min="1"
              max="120"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., 35"
              value={symptoms.height}
              onChange={(e) => setSymptoms(prev => ({ ...prev, height: parseInt(e.target.value) || 0 }))}
              required
            />
          </div>

          <div>
            <label htmlFor="weight" className="block text-sm font-medium text-gray-700 mb-2">
              Weight (kg)
            </label>
            <input
              id="weight"
              type="number"
              min="1"
              max="120"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., 35"
              value={symptoms.weight}
              onChange={(e) => setSymptoms(prev => ({ ...prev, weight: parseInt(e.target.value) || 0 }))}
              required
            />
          </div>

          <div>
            <label htmlFor="sex" className="block text-sm font-medium text-gray-700 mb-2">
              Biological Sex
            </label>
            <select
              id="sex"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white"
              value={symptoms.sex}
              onChange={(e) => setSymptoms(prev => ({ ...prev, sex: e.target.value }))}
              required
            >
              <option value="" disabled>Select...</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="unspecified">Prefer not to specify</option>
            </select>
          </div>
        </div>

        {error && (
          <div className="p-3 text-sm bg-red-100 border border-red-400 text-red-700 rounded-lg flex items-center">
            <Icon name="error" className="w-5 h-5 mr-2" />
            {error}
          </div>
        )}

        <button
          type="submit"
          className={`w-full flex items-center justify-center px-6 py-3 border border-transparent text-lg font-bold rounded-lg shadow-lg transition duration-200 ease-in-out transform hover:scale-[1.01] ${isFormValid && !loading ? 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-xl' : 'bg-gray-300 text-gray-600 cursor-not-allowed'}`}
          disabled={!isFormValid || loading}
        >
          {loading ? (
            <LoadingSpinner message="Analyzing..." />
          ) : (
            <>
              Get Analysis
              <Icon name="send" className="w-5 h-5 ml-3" />
            </>
          )}
        </button>
      </form>
    </div>
  );
};

/**
 * Component to display the diagnosis result.
 */
const DiagnosisResult = ({ diagnosis, symptoms, onNewCheck }) => {
  
  return (
    <div className="p-4 sm:p-8 bg-white rounded-xl shadow-2xl w-full max-w-3xl mx-auto">
      <h2 className="text-3xl font-extrabold text-green-700 mb-2 flex items-center">
        <Icon name="check" className="w-8 h-8 mr-2 text-green-500" />
        Health Analysis Complete
      </h2>
      <p className="text-gray-600 mb-6 border-b pb-4">
        Based on your input (Age: {symptoms.age}, Sex: {symptoms.sex}), here is the diagnosis.
      </p>

      {/* Primary Diagnosis Output */}
      <div className="p-6 border border-gray-200 rounded-lg bg-gray-50 shadow-inner mb-6">
        {/* Note: diagnosis.text contains markdown, which will be rendered safely */}
        <div className="prose max-w-none text-gray-800" dangerouslySetInnerHTML={{ __html: diagnosis.text }} />
      </div>

      <div className="p-6 border border-blue-200 rounded-lg bg-blue-50 shadow-inner mb-6">
        <p>And here are the AI generated recommendations:</p>
        <h3 className="text-xl font-bold text-blue-700 mb-2">Gemini Insights</h3>
        <div className="prose max-w-none text-blue-800" dangerouslySetInnerHTML={{ __html: diagnosis.gemini }} />
      </div>
      
      {/* Quick Actions */}
      <div className="flex justify-center">
        <button
          onClick={onNewCheck}
          className="flex items-center justify-center px-6 py-3 text-lg font-semibold rounded-lg shadow-md bg-blue-600 text-white hover:bg-blue-700 transition duration-200 transform hover:scale-[1.01]"
        >
          <Icon name="refresh" className="w-5 h-5 mr-2" />
          Start New Check
        </button>
      </div>

      
    </div>
  );
};


/**
 * Main Application Component (Exported by name 'App')
 */
export default function App() {
  const [symptoms, setSymptoms] = useState({
    symptomText: '',
    age: 30,
    sex: 'male',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [view, setView] = useState('form'); // 'form', 'loading', 'result'
  const [diagnosis, setDiagnosis] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [userId, setUserId] = useState(null);

  const callGeminiApi = useGeminiApi();



  // 2. Main function to call Gemini for Diagnosis
  const handleSymptomSubmission = useCallback(async () => {
    setError(null);
    setLoading(true);
    setView('loading');
    setDiagnosis(null);

    const userQuery = `Symptoms: ${symptoms.symptomText}. Age: ${symptoms.age}. Biological Sex: ${symptoms.sex}. Please analyze this.`;
    const systemPrompt = "You are a highly empathetic and knowledgeable AI symptom analyzer. Your primary goal is to provide a brief, clear, and non-diagnostic summary of the top 3 most *possible* conditions based *only* on the provided symptoms, age, and biological sex. For each condition, provide the confidence score (out of 100), key associated symptoms, and, most importantly, clear, immediate, and safe **recommendations/next steps** (e.g., self-care, monitor, see a doctor within 24 hours, go to the ER). **Crucially, start your response with a clear, empathetic disclaimer that you are not a doctor and cannot provide a medical diagnosis, use Markdown formatting for bolding and lists, and avoid using HTML tags like <h1> or <h2>, preferring standard Markdown text.**";
    try {
      const backendResponse = await fetch('http://localhost:5000/api/diagnosis', { // Replace /api/diagnosis with your actual endpoint
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
              symptoms: symptoms.symptomText, 
              age: symptoms.age, 
              sex: symptoms.sex 
          })
      });
      if (!backendResponse.ok) throw new Error('Backend failed to process diagnosis.');
      const diagnosisResult = await backendResponse.json(); // This result comes from checker.py
      setDiagnosis(diagnosisResult);
      setView('result');
    } catch (error) {
        console.error("Backend Error:", error);
        setError("Could not connect to the diagnosis service.");
        setView('form');
        setLoading(false);
        return;
    }
    setLoading(false);
  }, [symptoms, userId]);

  // Function to reset the application state
  const handleNewCheck = () => {
    setSymptoms({ symptomText: '', age: 30, sex: 'male' });
    setDiagnosis(null);
    setView('form');
    setError(null);
  };

  const currentView = useMemo(() => {
    if (loading) {
      return <LoadingSpinner message="Searching medical data and generating analysis..." />;
    }

    if (view === 'result' && diagnosis) {
      return (
        <DiagnosisResult
          diagnosis={diagnosis}
          symptoms={symptoms}
          onNewCheck={handleNewCheck}
        />
      );
    }

    return (
      <SymptomCheckerForm
        symptoms={symptoms}
        setSymptoms={setSymptoms}
        onSubmit={handleSymptomSubmission}
        loading={loading}
        error={error}
      />
    );
  }, [view, loading, diagnosis, symptoms, handleSymptomSubmission, handleNewCheck, error]);


  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-4 pt-10 font-sans">
      <header className="mb-8 w-full max-w-2xl text-center">
        <h1 className="text-4xl font-black text-blue-900 flex items-center justify-center">
          <Icon name="home" className="w-8 h-8 mr-2 text-blue-600" />
          HealthFlow AI
        </h1>
        <p className="text-sm text-gray-500 mt-1">Your AI-Powered Symptom and Health Assistant</p>
        <p className="text-xs text-gray-400 mt-2">User ID: <span className="font-mono text-gray-500">{userId || 'Loading...'}</span></p>
      </header>
      
      {currentView}

      <DisclaimerBanner />
    </div>
  );
}
