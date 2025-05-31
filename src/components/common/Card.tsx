
import React from 'react';

// Update CardProps to extend React.HTMLAttributes<HTMLDivElement>
interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  hoverEffect?: boolean;
  // className, onClick, and style will be covered by React.HTMLAttributes<HTMLDivElement>
}

export const Card: React.FC<CardProps> = ({ 
  children, 
  className = '', 
  hoverEffect = false, 
  onClick, // Explicitly get onClick to manage cursor style if needed
  ...rest // Spread the rest of the props (including style, id, etc.)
}) => {
  const baseStyles = 'bg-earthy-brown/30 backdrop-blur-sm p-6 rounded-xl shadow-xl';
  const hoverStyles = hoverEffect ? 'transform hover:scale-105 transition-transform duration-300 ease-out' : '';
  const clickableStyles = onClick ? 'cursor-pointer' : '';
  
  // Combine all class names, ensuring no extra spaces
  const combinedClassName = `${baseStyles} ${hoverStyles} ${clickableStyles} ${className}`.trim().replace(/\s+/g, ' ');

  return (
    <div
      className={combinedClassName}
      onClick={onClick} // Pass onClick explicitly
      {...rest} // Spread the rest of the props, including 'style'
    >
      {children}
    </div>
  );
};
