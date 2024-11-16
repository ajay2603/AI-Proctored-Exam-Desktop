function App() {
  const ipcHandle = () => window.electron.ipcRenderer.send('ping')

  return (
    <>
      <h1 className="text-4xl font-semibold text-center">AI Proctored Exam Desktop</h1>
    </>
  )
}

export default App
