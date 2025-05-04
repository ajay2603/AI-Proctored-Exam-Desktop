import { ipcMain } from 'electron'

export default function handleRenderer() {
  ipcMain.on('on-renderer', (event, data) => {
    console.log(data)
  })
}
