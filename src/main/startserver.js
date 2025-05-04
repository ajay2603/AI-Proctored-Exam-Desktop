import path from 'path'
import { spawn } from 'child_process'
import { app, ipcMain } from 'electron'
import mainWindow from './index'

var serverRunning = false
var localServer = null

const startServer = () => {
  let resourcesPath = app.isPackaged
    ? path.join(process.resourcesPath) // Use process.resourcesPath in production
    : path.join(app.getAppPath(), 'resources') // Use app.getAppPath() in development

  console.log('Resources Path:', resourcesPath)

  const serverSRC = path.join(resourcesPath, 'server')
  const python = path.join(serverSRC, 'venv', 'Scripts', 'python.exe')
  const serverScript = path.join(serverSRC, 'app.py')

  console.log('Python Path:', python)
  console.log('Server Script:', serverScript)

<<<<<<< HEAD
  localServer = spawn(python, [serverScript], { shell: true, cwd: serverSRC })
=======
  localServer = spawn(python, [serverScript], { cwd: serverSRC })
>>>>>>> 6ad55d2 (.)

  localServer.stdout.on('data', (data) => {
    serverRunning = true
    if (mainWindow)
      mainWindow.webContents.send('on-main', { event: 'server-stat', running: serverRunning })
    console.log(`stdout: ${data}`)
  })

  localServer.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`)
  })

  localServer.on('close', (code) => {
    serverRunning = false
    localServer = null
    if (mainWindow)
      mainWindow.webContents.send('on-main', { event: 'server-stat', running: serverRunning })
    console.log(`Python server exited with code ${code}`)
  })
}

const stopServer = () => {
  if (localServer) {
    console.log('Stopping Python server...');

    if (process.platform === 'win32') {
      console.log("win")
      // Get the Process ID (PID) of the server and kill it
      spawn('taskkill', ['/PID', localServer.pid, '/F', '/T'], { shell: true });
    } else {
      // For macOS/Linux, use normal kill
      console.log("lin")
      localServer.kill('SIGTERM');
    }

    localServer = null;
    console.log("Server stopped");
  }
};


app.whenReady().then(() => {
  ipcMain.on('on-renderer', (_, data) => {
    console.log(data)
    if (data.event === 'get-server-stat') {
      if (mainWindow)
        mainWindow.webContents.send('on-main', { event: 'server-stat', running: serverRunning })
    }
  })
  startServer()
})

// Handle app closing
app.on('before-quit', () => {
  stopServer()
})

// Handle window close (if using single-window mode)
app.on('window-all-closed', () => {
  stopServer()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
