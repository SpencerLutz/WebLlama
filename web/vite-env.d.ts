/// <reference types="vite/client" />

declare module '*.wgsl?raw' { // try removing to line 11
    const src: string;
    export default src;
}

declare module '*.html?raw' {
    const src: string;
    export default src;
}

declare module '*.module.css' {
    const classes: { [key: string]: string };
    export default classes;
}