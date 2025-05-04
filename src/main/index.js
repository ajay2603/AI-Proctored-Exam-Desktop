import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import './startserver'
import './terminate_apps'

let mainWindow = null
let examToken = null

import handleRenderer from './ipchandler'

handleRenderer()

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.webContents.openDevTools()

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  mainWindow.loadURL('http://localhost:5173/app')
}

app.setAsDefaultProtocolClient('ai-exam-app')

function handleProtocolUrl(url) {
  try {
    const token = new URL(url).searchParams.get('token')
    if (token) {
      examToken = token
      if (mainWindow) {
        mainWindow.webContents.send('on-main', { event: 'token', token: token })
      }
    }
  } catch (error) {
    console.error('Protocol URL error:', error)
  }
}

ipcMain.on('on-renderer', (_, data) => {
  console.log(data)
  if (data.event == 'get-token') {
    mainWindow.webContents.send('on-main', { event: 'token', token: examToken })
  }
})

// Single instance lock
if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', (event, commandLine) => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()

      const protocolUrl = commandLine.find((arg) => arg.startsWith('ai-exam-app://'))
      if (protocolUrl) handleProtocolUrl(protocolUrl)
    }
  })

  app.whenReady().then(() => {
    // Set app user model id for windows
    electronApp.setAppUserModelId('com.electron')

    // Default open or close DevTools by F12 in development
    // and ignore CommandOrControl + R in production.
    // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
    app.on('browser-window-created', (_, window) => {
      optimizer.watchWindowShortcuts(window)
    })

    // IPC test
    ipcMain.on('ping', () => console.log('pong'))

    createWindow()

    app.on('activate', function () {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })

    const protocolUrl = process.argv.find((arg) => arg.startsWith('ai-exam-app://'))
    if (protocolUrl) handleProtocolUrl(protocolUrl)
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

ipcMain.on('on-renderer', (_, data) => {
  if (data.event === 'pre-exam') {
    mainWindow.setKiosk(true)
  }

  if (data.event === 'post-exam') {
    mainWindow.setKiosk(false)
  }

<<<<<<< HEAD
<<<<<<< HEAD
  if (data.evet === 'quit-app') {
=======
  if (data.event === 'quit-app') {
>>>>>>> 6ad55d2 (.)
=======
  if (data.evet === 'quit-app') {
>>>>>>> 9e4669a0c9a0d75793787f6356d60ce7bb180376
    app.quit()
  }
})

// In this file you can include the rest of your app"s specific main process
// code. You can also put them in separate files and require them here.
export default mainWindow
