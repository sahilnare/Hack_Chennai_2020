import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { ZeitProvider, CssBaseline } from '@zeit-ui/react'

ReactDOM.render(
    <ZeitProvider>
      <CssBaseline />
      <App />
    </ZeitProvider>,
  document.getElementById('root')
);
