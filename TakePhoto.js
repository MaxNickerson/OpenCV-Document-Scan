import React, { useRef, useState } from 'react';
import Camera from 'react-camera-pro';

const TakePhoto = () => {
    const camera = useRef(null);
    const [image, setImage] = useState(null);

    const handleTakePhoto = () => {
        if (camera.current) {
            const photo = camera.current.takePhoto();
            setImage(photo);
        }
    };

    const handleSendPhoto = async () => {
        if (!image) return alert("No photo taken!");
        const formData = new FormData();
        formData.append("photo", dataURItoBlob(image)); // Convert Base64 to Blob

        try {
            const response = await fetch('/api/upload-photo', {
                method: 'POST',
                body: formData,
            });
            if (response.ok) {
                alert("Photo uploaded successfully!");
            } else {
                alert("Upload failed.");
            }
        } catch (error) {
            console.error(error);
            alert("Error uploading photo.");
        }
    };

    // Helper to convert data URI to Blob
    const dataURItoBlob = (dataURI) => {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const buffer = new ArrayBuffer(byteString.length);
        const view = new Uint8Array(buffer);
        for (let i = 0; i < byteString.length; i++) {
            view[i] = byteString.charCodeAt(i);
        }
        return new Blob([buffer], { type: mimeString });
    };

    return (
        <div>
            <Camera ref={camera} aspectRatio={16 / 9} />
            <button onClick={handleTakePhoto}>Take Photo</button>
            {image && <img src={image} alt="Captured" />}
            <button onClick={handleSendPhoto}>Send Photo</button>
        </div>
    );
};

export default TakePhoto;